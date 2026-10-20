/* gr_info: read-only GB10 inventory through our own path (physics nvrm, open nvidia.ko RM control).
 * No channel, no allocation, no launch: one RM client, one NV2080_CTRL_CMD_GR_GET_INFO, close.
 * Index meanings: physics/third_party/nvidia-open-580.173.02/src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080gr.h.
 * Build: see build.sh next to this file (PHYSICS_DIR pinned to physics.lock 6d7cf0d4). */
#include "nvrm.h"
#include "nvos.h"
#include "nv-ioctl.h"
#include "nv-ioctl-numbers.h"
#include "nv_escape.h"
#include "ctrl/ctrl2080/ctrl2080gr.h"

#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>

/* same definition as physics nvrm/nvrm.c:48 and forge/forge_descriptor.c:20 */
#define NV_IOWR(nr, sz) _IOC(_IOC_READ | _IOC_WRITE, NV_IOCTL_MAGIC, (nr), (sz))

#define Q(x) { NV2080_CTRL_GR_INFO_INDEX_##x, #x }
static const struct { uint32_t index; const char *name; } want[] = {
    Q(SM_VERSION), Q(SHADER_PIPE_COUNT), Q(SHADER_PIPE_SUB_COUNT), Q(LITTER_NUM_GPCS),
    Q(LITTER_NUM_TPC_PER_GPC), Q(LITTER_NUM_SM_PER_TPC), Q(MAX_WARPS_PER_SM),
    Q(MAX_THREADS_PER_WARP), Q(MAX_SP_PER_SM), Q(GPU_CORE_COUNT), Q(SM_REG_BANK_COUNT),
    Q(SM_REG_BANK_REG_COUNT), Q(LITTER_NUM_LTCS), Q(LITTER_NUM_LTC_SLICES),
    Q(LITTER_NUM_SLICES_PER_LTC), Q(LITTER_NUM_FBPS),
};
#define N (sizeof want / sizeof want[0])

int main(void) {
    Nvrm rm;
    memset(&rm, 0, sizeof rm);
    if (nvrm_open(&rm)) { fprintf(stderr, "nvrm_open failed\n"); return 1; }
    NV2080_CTRL_GR_INFO list[N];
    memset(list, 0, sizeof list);
    for (size_t i = 0; i < N; i++) list[i].index = want[i].index;
    NV2080_CTRL_GR_GET_INFO_PARAMS gp;
    memset(&gp, 0, sizeof gp);
    gp.grInfoListSize = N;
    gp.grInfoList = (NvP64)(uintptr_t)list;
    NVOS54_PARAMETERS p;
    memset(&p, 0, sizeof p);
    p.hClient = rm.root;
    p.hObject = rm.subdevice;
    p.cmd = NV2080_CTRL_CMD_GR_GET_INFO;
    p.params = (NvP64)(uintptr_t)&gp;
    p.paramsSize = sizeof gp;
    int rc = ioctl(rm.fd_ctl, NV_IOWR(NV_ESC_RM_CONTROL, sizeof p), &p);
    if (rc != 0 || p.status != 0) {
        fprintf(stderr, "GR_GET_INFO rc %d status 0x%x\n", rc, p.status);
        nvrm_close(&rm);
        return 1;
    }
    printf("compute_class 0x%x\n", rm.compute_class);
    for (size_t i = 0; i < N; i++) printf("%s %u\n", want[i].name, list[i].data);
    nvrm_close(&rm);
    return 0;
}
