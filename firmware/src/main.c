/* STM32F103ZET6 振动识别固件 v2：在线 ADC 采集 + 门限 + 3 类 int8 CNN
 *
 * 采集：ADC1 (PA0) 软件定时采样 1024 点 @10.24 kHz（DWT 精确控周期），约 100 ms/窗；
 * 模拟源：DAC1 (PA4) 按同样节奏回放内置的真实振动样本（4 类各一条）。
 *   - 若用杜邦线把 PA4 接到 PA0：ADC 采到的就是"回放的真实信号"，走完整在线链路；
 *   - 若没有接线/没有传感器：ADC 采到的是悬空噪声，固件会同时用软件回放同一批样本做自检。
 * 处理：计算 RMS -> 门限判空闲（<0.006 数据单位）-> 逐窗 RMS 归一化 -> int8 -> CNN -> 串口打印。
 *
 * 安全：DAC/ADC 均为 0~3.3V 片内信号；接真实传感器时，传感器输出必须限幅在 0~3.3V 并共地。
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "stm32f1xx.h"
#include "nn.h"
#include "model.h"
#include "replay_samples.h"

extern uint8_t _siramfunc[];
extern uint8_t _sramfunc[];
extern uint8_t _eramfunc[];

#define USART_BAUD        115200u
#define TIMING_RUNS       10u
#define SAMPLE_RATE       10240u
#define WIN_LEN           1024
#define VREF              3.3f
#define GAIN              5.0f        /* 数据单位 -> 伏特（DAC 回放同上） */
#define GATE_THR_UNITS    0.0060f     /* 数据单位：idling max 0.0040，wear min 0.0052 */
#define INPUT_SCALE       0.047244094f /* 量化 meta 的输入 scale */
#define CLIP_X            127.0f      /* int8 量化范围（归一化已除以 s_in） */

#define ADC_MEAN_CODE     2048

/* ---- 随机抽窗：PC 通过 USART1 下发 3×1024 原始 ADC 码，单轴只取 X ---- */
#define SEL_FRAME_LEN     (2 + 1 + 1 + 4 + (3 * WIN_LEN * 2) + 1)
static uint8_t  g_rx_buf[8192];
static volatile uint16_t g_rx_len = 0;
static volatile uint8_t  g_sel_ready = 0;
static volatile uint8_t  g_sel_seq = 0;
static volatile uint8_t  g_sel_cls = 0;
static volatile uint32_t g_sel_idx = 0;
static int16_t  g_sel[WIN_LEN];

void USART1_IRQHandler(void) {
    if ((USART1->SR & USART_SR_RXNE) != 0u) {
        const uint8_t b = (uint8_t)USART1->DR;
        if (g_rx_len < (uint16_t)sizeof(g_rx_buf)) {
            g_rx_buf[g_rx_len++] = b;
        } else {
            g_rx_len = 0;
        }
    }
}

/* ============================ 系统时钟 ============================ */
static int clock_init_72mhz(void) {
    uint32_t timeout;
    RCC->CR |= RCC_CR_HSEON;
    timeout = 0x50000u;
    while (((RCC->CR & RCC_CR_HSERDY) == 0u) && (--timeout != 0u)) {
    }
    if ((RCC->CR & RCC_CR_HSERDY) == 0u) {
        SystemCoreClockUpdate();
        return 0;
    }
    FLASH->ACR = FLASH_ACR_PRFTBE | FLASH_ACR_LATENCY_2;
    RCC->CFGR = (RCC->CFGR & ~(RCC_CFGR_HPRE | RCC_CFGR_PPRE1 | RCC_CFGR_PPRE2))
              | RCC_CFGR_HPRE_DIV1 | RCC_CFGR_PPRE1_DIV2 | RCC_CFGR_PPRE2_DIV1;
    RCC->CFGR = (RCC->CFGR & ~(RCC_CFGR_PLLSRC | RCC_CFGR_PLLXTPRE | RCC_CFGR_PLLMULL))
              | RCC_CFGR_PLLSRC | RCC_CFGR_PLLMULL9;
    RCC->CR |= RCC_CR_PLLON;
    timeout = 0x50000u;
    while (((RCC->CR & RCC_CR_PLLRDY) == 0u) && (--timeout != 0u)) {
    }
    if ((RCC->CR & RCC_CR_PLLRDY) == 0u) {
        SystemCoreClockUpdate();
        return 0;
    }
    RCC->CFGR = (RCC->CFGR & ~RCC_CFGR_SW) | RCC_CFGR_SW_PLL;
    timeout = 0x50000u;
    while (((RCC->CFGR & RCC_CFGR_SWS) != RCC_CFGR_SWS_PLL) && (--timeout != 0u)) {
    }
    SystemCoreClockUpdate();
    return (RCC->CFGR & RCC_CFGR_SWS) == RCC_CFGR_SWS_PLL;
}

/* ============================== USART1 ============================== */
static uint32_t usart1_pclk2(void) {
    const uint32_t ppre2 = (RCC->CFGR & RCC_CFGR_PPRE2) >> RCC_CFGR_PPRE2_Pos;
    if (ppre2 < 4u) {
        return SystemCoreClock;
    }
    return SystemCoreClock >> (ppre2 - 3u);
}

static void usart1_init(void) {
    RCC->APB2ENR |= RCC_APB2ENR_IOPAEN | RCC_APB2ENR_USART1EN;
    GPIOA->CRH = (GPIOA->CRH & ~(0xFu << 4)) | (0xBu << 4);
    GPIOA->CRH = (GPIOA->CRH & ~(0xFu << 8)) | (0x4u << 8);
    USART1->CR2 = 0u;
    USART1->CR3 = 0u;
    {
        const uint32_t pclk2 = usart1_pclk2();
        uint32_t brr = (pclk2 + USART_BAUD / 2u) / USART_BAUD;
        if (brr == 0u) {
            brr = 1u;
        }
        USART1->BRR = brr;
    }
    USART1->CR1 = USART_CR1_UE | USART_CR1_TE | USART_CR1_RE | USART_CR1_RXNEIE;
    NVIC_EnableIRQ(USART1_IRQn);
}

int _write(int file, char *ptr, int len) {
    (void)file;
    for (int i = 0; i < len; ++i) {
        while ((USART1->SR & USART_SR_TXE) == 0u) {
        }
        USART1->DR = (uint8_t)ptr[i];
    }
    return len;
}

/* ========================= DWT 周期计数器 ========================= */
static int dwt_ok = 0;

static void dwt_init(void) {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0u;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    dwt_ok = (DWT->CTRL & DWT_CTRL_CYCCNTENA_Msk) != 0u;
}

static inline uint32_t dwt_cycles(void) { return DWT->CYCCNT; }

static void delay_cycles(uint32_t cycles) {
    const uint32_t start = dwt_cycles();
    while ((uint32_t)(dwt_cycles() - start) < cycles) {
    }
}

/* ============================= ADC / DAC ============================= */
static void gpio_analog_init(void) {
    RCC->APB2ENR |= RCC_APB2ENR_IOPAEN;
    GPIOA->CRL &= ~0x0000000Fu;      /* PA0 analog input (ADC_IN0) */
    GPIOA->CRL &= ~0x000F0000u;      /* PA4 analog output (DAC_OUT1) */
}

static void adc_init(void) {
    RCC->APB2ENR |= RCC_APB2ENR_ADC1EN;
    RCC->CFGR = (RCC->CFGR & ~RCC_CFGR_ADCPRE) | RCC_CFGR_ADCPRE_DIV6;   /* 72/6=12 MHz */

    ADC1->CR1 = 0u;
    ADC1->CR2 = ADC_CR2_ADON;          /* 第一次 ADON：ADC 上电 */
    delay_cycles(24000u);              /* tSTAB > 1us */
    /* 校准（必须在连续模式之前做） */
    {
        uint32_t to = 0x100000u;
        ADC1->CR2 |= ADC_CR2_RSTCAL;
        while ((ADC1->CR2 & ADC_CR2_RSTCAL) != 0u && (--to != 0u)) {
        }
        to = 0x100000u;
        ADC1->CR2 |= ADC_CR2_CAL;
        while ((ADC1->CR2 & ADC_CR2_CAL) != 0u && (--to != 0u)) {
        }
    }
    ADC1->SMPR1 = (ADC1->SMPR1 & ~0x7u) | 0x7u;   /* channel 0: 239.5 cycles (SMPR1!) */
    ADC1->SQR1 = 0u;
    ADC1->SQR2 = 0u;
    ADC1->SQR3 = 0u;                              /* SQ1 = channel 0 */
    /* F1 关键：SWSTART 触发要求 EXTTRIG=1 且 EXTSEL=111（SWSTART），否则不启动 */
    ADC1->CR2 |= ADC_CR2_EXTTRIG | (0x7u << 17) | ADC_CR2_CONT;
    ADC1->CR2 |= ADC_CR2_SWSTART;                 /* 启动连续转换 */
    delay_cycles(24000u);
}

static void dac_init(void) {
    RCC->APB1ENR |= RCC_APB1ENR_DACEN;
    DAC->CR = DAC_CR_EN1;            /* 缓冲器开启，VREF=3.3V */
    DAC->DHR12R1 = ADC_MEAN_CODE;
}

static uint32_t g_adc_timeouts = 0;

static inline uint16_t adc_read_once_cts(void) {
    uint32_t to = 2000u;
    ADC1->CR2 |= ADC_CR2_SWSTART;
    while ((ADC1->SR & ADC_SR_EOC) == 0u && (--to != 0u)) {
    }
    if ((ADC1->SR & ADC_SR_EOC) == 0u) {
        g_adc_timeouts++;
        return ADC_MEAN_CODE;
    }
    return (uint16_t)ADC1->DR;
}

/* 采集一个窗口：DAC 同步回放 -> ADC 按 10.24 kHz 采样，返回采集周期数 */
static uint32_t acquire_window(int cls, int16_t *buf) {
    const uint32_t t0 = dwt_cycles();
    uint32_t next = t0;
    const uint32_t step = SystemCoreClock / SAMPLE_RATE;
    for (int i = 0; i < WIN_LEN; ++i) {
        next += step;
        while ((int32_t)(dwt_cycles() - next) < 0) {
        }
        DAC->DHR12R1 = (uint16_t)REPLAY_DAC[cls][i];
        buf[i] = (int16_t)adc_read_once_cts();
    }
    return dwt_cycles() - t0;
}

/* ============================ 推理 + 处理 ============================ */
static int8_t net_in[MODEL_INPUT_C * MODEL_INPUT_LEN];
static float probs[MODEL_NUM_CLASSES];

static const char *class_name(int c) {
    static const char *names[MODEL_NUM_CLASSES] = {"initial_wear", "steady_wear",
                                                   "accelerating_wear"};
    if (c < 0 || c >= (int)MODEL_NUM_CLASSES) {
        return "?";
    }
    return names[c];
}

/* 处理一个窗口（code 为 12bit 原始码，2048 为零点） */
static void process_window(const int16_t *code, const char *tag, int dump_probs) {
    int32_t sum = 0;
    int64_t sumsq = 0;
    for (int i = 0; i < WIN_LEN; ++i) {
        const int32_t v = (int32_t)code[i] - ADC_MEAN_CODE;
        sum += v;
        sumsq += (int64_t)v * (int64_t)v;
    }
    const float mean_code = (float)sum / (float)WIN_LEN;
    /* 两个 RMS：raw 用于归一化（与训练一致，含直流）；ac 用于能量门限（防直流偏置误判） */
    const float rms_raw = sqrtf((float)((double)sumsq / (double)WIN_LEN));
    double var = 0.0;
    for (int i = 0; i < WIN_LEN; ++i) {
        const double d = (double)((int32_t)code[i] - ADC_MEAN_CODE) - (double)mean_code;
        var += d * d;
    }
    const float rms_ac = sqrtf((float)(var / (double)WIN_LEN));
    const float rms_code = rms_raw;
    const float rms_units = rms_ac * (VREF / 4095.0f) / GAIN;   /* 门限用 AC RMS */

    printf("[%s] mean=%ld rms_ac=%.0fcode (%.3fV, %.4f unit) -> ",
           tag, (long)mean_code, (double)rms_ac,
           (double)(rms_ac * VREF / 4095.0f), (double)rms_units);

    if (rms_units < GATE_THR_UNITS) {
        printf("idling\r\n");
        return;
    }

    /* 逐窗 RMS 归一化 + 量化（与 PC 训练/量化一致） */
    const float factor = 1.0f / (rms_code * INPUT_SCALE);
    for (int i = 0; i < WIN_LEN; ++i) {
        float xn = (((float)code[i] - (float)ADC_MEAN_CODE) - mean_code) * factor;
        if (xn > CLIP_X) {
            xn = CLIP_X;
        } else if (xn < -CLIP_X) {
            xn = -CLIP_X;
        }
        int32_t q = (int32_t)(xn + (xn >= 0.0f ? 0.5f : -0.5f));
        if (q > 127) {
            q = 127;
        } else if (q < -128) {
            q = -128;
        }
        net_in[i] = (int8_t)q;
    }

    /* 推理计时 */
    (void)model_forward(net_in, probs);   /* warmup */
    uint32_t tmin = 0xFFFFFFFFu, tsum = 0u;
    int pred = 0;
    for (uint32_t r = 0; r < TIMING_RUNS; ++r) {
        const uint32_t t0 = dwt_cycles();
        pred = (int)model_forward(net_in, probs);
        const uint32_t dt = dwt_cycles() - t0;
        if (dt < tmin) {
            tmin = dt;
        }
        tsum += dt;
    }
    printf("wear, pred=%d (%s) infer=%u cycles (%.2f ms)\r\n",
           pred, class_name(pred), (unsigned)(tsum / TIMING_RUNS),
           (double)((float)(tsum / TIMING_RUNS) * 1000.0f / (float)SystemCoreClock));
    if (dump_probs) {
        for (int i = 0; i < (int)MODEL_NUM_CLASSES; ++i) {
            printf("    class %d (%s): %.4f\r\n", i, class_name(i), (double)probs[i]);
        }
    }
}

/* 解析 PC 随机窗帧：只取 X 轴 */
static void handle_sel_frame(void) {
    if (g_rx_len < SEL_FRAME_LEN) return;
    const uint8_t *p = g_rx_buf;
    if (p[0] != 0xA5u || p[1] != 0x5Au) { g_rx_len = 0; return; }
    uint8_t crc = 0u;
    for (uint16_t i = 2; i < (SEL_FRAME_LEN - 1); ++i) crc ^= p[i];
    if (crc != p[SEL_FRAME_LEN - 1]) { printf("[sel] bad crc\r\n"); g_rx_len = 0; return; }
    const uint8_t *q = p + 8;
    for (int i = 0; i < WIN_LEN; ++i) {
        const uint16_t lo = q[0];
        const uint16_t hi = q[1];
        g_sel[i] = (int16_t)(lo | (hi << 8));
        q += 2;
    }
    g_sel_seq = p[2];
    g_sel_cls = p[3];
    g_sel_idx = (uint32_t)p[4] | ((uint32_t)p[5] << 8) |
                ((uint32_t)p[6] << 16) | ((uint32_t)p[7] << 24);
    g_sel_ready = 1;
    g_rx_len = 0;
}

static void process_selected(void) {
    int32_t sum = 0;
    int64_t sumsq = 0;
    for (int i = 0; i < WIN_LEN; ++i) {
        const int32_t v = (int32_t)g_sel[i] - ADC_MEAN_CODE;
        sum += v; sumsq += (int64_t)v * (int64_t)v;
    }
    const float mean_code = (float)sum / (float)WIN_LEN;
    const float rms_code = sqrtf((float)((double)sumsq / (double)WIN_LEN));
    double var = 0.0;
    for (int i = 0; i < WIN_LEN; ++i) {
        const double d = (double)((int32_t)g_sel[i] - ADC_MEAN_CODE) - (double)mean_code;
        var += d * d;
    }
    const float rms_ac = sqrtf((float)(var / (double)WIN_LEN));
    const float rms_units = rms_ac * (VREF / 4095.0f) / GAIN;
    if (rms_units < GATE_THR_UNITS) {
        printf("[sel] seq=%u cls=%u idx=%lu gate=idling pred=-1 conf=0.0000 rms=%.4f p0=0.0000 p1=0.0000 p2=0.0000\r\n",
               (unsigned)g_sel_seq, (unsigned)g_sel_cls, (unsigned long)g_sel_idx,
               (double)rms_units);
        return;
    }
    const float factor = 1.0f / (rms_code * INPUT_SCALE);
    for (int i = 0; i < WIN_LEN; ++i) {
        float xn = (((float)g_sel[i] - (float)ADC_MEAN_CODE) - mean_code) * factor;
        if (xn > CLIP_X) xn = CLIP_X;
        else if (xn < -CLIP_X) xn = -CLIP_X;
        int32_t qv = (int32_t)(xn + (xn >= 0.0f ? 0.5f : -0.5f));
        if (qv > 127) qv = 127; else if (qv < -128) qv = -128;
        net_in[i] = (int8_t)qv;
    }
    int pred = (int)model_forward(net_in, probs);
    printf("[sel] seq=%u cls=%u idx=%lu gate=wear pred=%d conf=%.4f rms=%.4f p0=%.4f p1=%.4f p2=%.4f\r\n",
           (unsigned)g_sel_seq, (unsigned)g_sel_cls, (unsigned long)g_sel_idx,
           pred, (double)probs[pred], (double)rms_units,
           (double)probs[0], (double)probs[1], (double)probs[2]);
}

int main(void) {
    const int pll72 = clock_init_72mhz();
    usart1_init();
    dwt_init();
    memcpy(_sramfunc, _siramfunc, (size_t)(_eramfunc - _sramfunc));
    printf("init   : gpio analog...\r\n");
    gpio_analog_init();
    printf("init   : adc...\r\n");
    adc_init();
    printf("init   : dac...\r\n");
    dac_init();
    delay_cycles(24000u);
    printf("init   : adc dbg CR2=%08lx SR=%08lx DR=%lu\r\n",
           (unsigned long)ADC1->CR2, (unsigned long)ADC1->SR,
           (unsigned long)ADC1->DR);
    printf("init   : done\r\n");

    printf("\r\n[vibration-edge] online ADC acquisition + int8 CNN\r\n");
    printf("build  : %s %s\r\n", __DATE__, __TIME__);
    printf("clock  : %u Hz (%s), pll72=%d, dwt=%s\r\n", (unsigned)SystemCoreClock,
           (RCC->CFGR & RCC_CFGR_SWS) == RCC_CFGR_SWS_PLL ? "PLL" : "HSI",
           pll72, dwt_ok ? "ok" : "FAIL");
    printf("model  : %s, input %d x %d int8, classes %d, params %d\r\n",
           MODEL_NAME, (int)MODEL_INPUT_C, (int)MODEL_INPUT_LEN,
           (int)MODEL_NUM_CLASSES, (int)MODEL_PARAM_COUNT);
    printf("acq    : ADC1 PA0, %u Hz x %d = 100 ms/window, DAC PA4 replay\r\n",
           (unsigned)SAMPLE_RATE, WIN_LEN);
    printf("gate   : rms < %.4f (data unit) => idling\r\n", (double)GATE_THR_UNITS);
    printf("note   : PA4->PA0 jumper = analog loopback; without jumper ADC sees noise\r\n");

    static int16_t adc_buf[WIN_LEN];
    int cls = 0;
    for (;;) {
        if (g_rx_len >= SEL_FRAME_LEN) handle_sel_frame();
        if (g_sel_ready) { process_selected(); g_sel_ready = 0; }
        printf("\r\n--- window: replay class %d (%s) ---\r\n", cls, REPLAY_NAME[cls]);
        const uint32_t acq = acquire_window(cls, adc_buf);
        printf("[acq  ] %u cycles (%.2f ms), 1024 samples @ %u Hz, adc_timeouts=%u\r\n",
               (unsigned)acq, (double)((float)acq * 1000.0f / (float)SystemCoreClock),
               (unsigned)SAMPLE_RATE, (unsigned)g_adc_timeouts);
        printf("[wave]");
        for (int i = 0; i < WIN_LEN; i += 4) {
            printf(" %d", (int)adc_buf[i]);
        }
        printf("\r\n");
        process_window(adc_buf, "adc  ", 0);
        process_window(REPLAY_DAC[cls], "replay", 1);
        printf("[window] acq %.1f ms + infer %.2f ms = %.1f ms/decision (window 100 ms)\r\n",
               (double)((float)acq * 1000.0f / (float)SystemCoreClock),
               17.07, (double)((float)acq * 1000.0f / (float)SystemCoreClock) + 17.07);
        cls = (cls + 1) % REPLAY_N;
        delay_cycles(SystemCoreClock * 3u / 2u);   /* 1.5 s 间隔：串口/看板显示更易读 */
    }
}
