#ifndef OMX_CONF_ESD_LINK_TASK_H
#define OMX_CONF_ESD_LINK_TASK_H

/* Built-in protocol stress endpoints. Set to 0 for production images. */
#ifndef OMX_ESD_V1_ENABLE_TEST
#define OMX_ESD_V1_ENABLE_TEST 1
#endif

#define OMX_ESD_LINK_TEST_CONTROL_CHANNEL 0u

/* Motor control and host feedback share this project cadence. */
#define OMX_ESD_CONTROL_RATE_HZ 500u

#endif /* OMX_CONF_ESD_LINK_TASK_H */
