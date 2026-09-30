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
// Independent parameters: armature(6), damping(6), frictionloss(6), delay(6).
// Outputs: q(6), dq(6), commanded torque(6), scale(6), interval peak contacts.
int pe_run_independent(void* ptr, const double* par, double dt, double control_dt,
           int n, const double* times, const double* commands,
           const double* q0, const double* dq0, int ns,
           const double* sample_times, double* out, int legacy_clip, int contacts) {
  auto* r = static_cast<Replay*>(ptr); auto* m = r->m; auto* d = r->d;
  m->opt.timestep = dt;
  if (contacts) m->opt.disableflags &= ~mjDSBL_CONTACT;
  else m->opt.disableflags |= mjDSBL_CONTACT;
  for (int j=0; j<6; ++j) {
    m->dof_armature[j] = par[j]; m->dof_damping[j] = par[6+j];
    m->dof_frictionloss[j] = par[12+j];
  }
  mj_resetData(m,d);
  std::memcpy(d->qpos,q0,6*sizeof(double));
  std::memcpy(d->qvel,dq0,6*sizeof(double));
  mj_forward(m,d);
  int idx[6]={}; int s=0; double next_control=0;
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
      for (int j=0;j<6;++j)
        while (idx[j]+1<n && times[idx[j]+1]<=d->time-par[18+j]+1e-9) ++idx[j];
      for (int j=0;j<6;++j) {
        const double* c=commands+idx[j]*30;
        double p=c[12+j]*(c[j]-d->qpos[j]);
        double v=c[18+j]*(c[6+j]-d->qvel[j]);
        double demand=std::abs(p)+std::abs(v)+std::abs(c[24+j]);
        scale[j] = demand>limits[j%3] ? limits[j%3]/demand : 1.0;
      }
      next_control += control_dt;
    }
    for (int j=0;j<6;++j) {
      const double* c=commands+idx[j]*30;
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
// Preserve the original API and exact grouped-parameter semantics.
int pe_run(void* ptr, const double* par, double dt, double control_dt,
           int n, const double* times, const double* commands,
           const double* q0, const double* dq0, int ns,
           const double* sample_times, double* out, int legacy_clip, int contacts) {
  double independent[24];
  for (int block=0; block<4; ++block)
    for (int j=0; j<6; ++j) independent[block*6+j]=par[block*3+j%3];
  return pe_run_independent(ptr, independent, dt, control_dt, n, times,
                            commands, q0, dq0, ns, sample_times, out, legacy_clip, contacts);
}
}
