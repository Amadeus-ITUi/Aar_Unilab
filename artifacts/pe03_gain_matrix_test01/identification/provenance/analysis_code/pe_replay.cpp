// Offline PE fixture replay. No hardware or production control dependencies.
#include <mujoco/mujoco.h>
#include <algorithm>
#include <cmath>
#include <cstring>

struct Replay { mjModel* m; mjData* d; };
extern "C" {
void* pe_create(const char* filename) {
  mjModel* m = mj_loadModel(filename, nullptr);
  if (!m || m->nq != 6 || m->nv != 6) { if (m) mj_deleteModel(m); return nullptr; }
  return new Replay{m, mj_makeData(m)};
}
void pe_destroy(void* ptr) {
  auto* r = static_cast<Replay*>(ptr);
  if (r) { mj_deleteData(r->d); mj_deleteModel(r->m); delete r; }
}
// Commands: row-major [q(6), dq(6), kp(6), kd(6), ff(6)].
// Parameters: total armature(3), passive damping(3), frictionloss(3), delay(3).
// Outputs: q(6), dq(6), commanded torque(6), scale(6), interval peak contacts.
int pe_run(void* ptr, const double* par, double dt, double control_dt,
           int n, const double* times, const double* commands,
           const double* q0, const double* dq0, int ns,
           const double* sample_times, double* out, int legacy_clip, int contacts) {
  auto* r = static_cast<Replay*>(ptr); auto* m = r->m; auto* d = r->d;
  m->opt.timestep = dt;
  if (contacts) m->opt.disableflags &= ~mjDSBL_CONTACT;
  else m->opt.disableflags |= mjDSBL_CONTACT;
  for (int j=0; j<6; ++j) {
    m->dof_armature[j] = par[j%3]; m->dof_damping[j] = par[3+j%3];
    m->dof_frictionloss[j] = par[6+j%3];
  }
  mj_resetData(m,d);
  std::memcpy(d->qpos,q0,6*sizeof(double));
  std::memcpy(d->qvel,dq0,6*sizeof(double));
  mj_forward(m,d);
  int idx[3]={0,0,0}; int s=0; double next_control=0;
  int peak_contacts=0;
  double scale[6]={1,1,1,1,1,1}, torque[6]={};
  const double limits[3]={5.5,5.5,14};
  while (s<ns) {
    while (s<ns && sample_times[s]<=d->time+1e-9) {
      for (int j=0;j<6;++j) {
        out[s*25+j]=d->qpos[j]; out[s*25+6+j]=d->qvel[j];
        out[s*25+12+j]=torque[j]; out[s*25+18+j]=scale[j];
      }
      out[s*25+24]=std::max(peak_contacts,d->ncon); peak_contacts=0; ++s;
    }
    if (s==ns) break;
    if (d->time+1e-9>=next_control) {
      for (int g=0;g<3;++g)
        while (idx[g]+1<n && times[idx[g]+1]<=d->time-par[9+g]+1e-9) ++idx[g];
      for (int j=0;j<6;++j) {
        const double* c=commands+idx[j%3]*30;
        double p=c[12+j]*(c[j]-d->qpos[j]);
        double v=c[18+j]*(c[6+j]-d->qvel[j]);
        double demand=std::abs(p)+std::abs(v)+std::abs(c[24+j]);
        scale[j] = demand>limits[j%3] ? limits[j%3]/demand : 1.0;
      }
      next_control += control_dt;
    }
    for (int j=0;j<6;++j) {
      const double* c=commands+idx[j%3]*30;
      double raw=c[12+j]*(c[j]-d->qpos[j])+c[18+j]*(c[6+j]-d->qvel[j])+c[24+j];
      torque[j]=legacy_clip ? std::clamp(raw,-limits[j%3],limits[j%3]) : scale[j]*raw;
      d->qfrc_applied[j]=torque[j];
    }
    mj_step(m,d);
    peak_contacts=std::max(peak_contacts,d->ncon);
    for(int j=0;j<6;++j) if (!std::isfinite(d->qpos[j]) || std::abs(d->qvel[j])>1e4) return -1;
    if (d->warning[mjWARN_BADQPOS].number || d->warning[mjWARN_BADQVEL].number ||
        d->warning[mjWARN_BADQACC].number) return -2;
  }
  return 0;
}
}
