uint8 POSITIONING=0
uint8 POSITION_HOLD=1
uint8 RC_CONTROL=2
uint8 FAULT=3

builtin_interfaces/Time stamp
uint8 state
int32[] motor_ids
float32[] position
float32[] velocity
bool feedback_ok
string fault_code
string message
