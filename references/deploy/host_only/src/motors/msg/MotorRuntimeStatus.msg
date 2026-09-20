uint8 DISARMED=0
uint8 STANDBY=1
uint8 ACTIVE=2
uint8 FAULT=3

builtin_interfaces/Time stamp
uint8 state
int32[] motor_ids
bool[] online
float32[] position
float32[] velocity
float32[] effort
string fault_code
string message
