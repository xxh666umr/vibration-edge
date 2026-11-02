/* STM32F103ZET6 三轴振动识别固件（x/y/z 联合输入，int8 CNN，离线回放演示）
 *
 * 采集：DAC1(PA4) 分时回放三轴真实样本 -> ADC1(PA0) 以 10.24 kHz 采样；
 *      每个窗口 = 3 轴 × 1024 点，用时 3 × 100 ms = 300 ms。
 * 处理：三轴联合 AC-RMS 门限判空闲 -> 联合归一化 -> int8 3 类 CNN。
 * 串口：USART1 (PA9/PA10) 115200 8N1。
 *
 * 若把 PA4->PA0 短接，ADC 采到的就是回放波形；不接则 ADC 为悬空噪声，门限判空闲。
 * 接真实三轴传感器时：三个通道分别接 PA0/PA1/PA2（0~3.3V，共地），并改用 ADC 扫描模式。
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "stm32f1xx.h"
#include "nn.h"
#include "model.h"
#include "replay_xyz.h"

extern uint8_t _siramfunc[];
extern uint8_t _sramfunc[];
extern uint8_t _eramfunc[];

#define USART_BAUD        115200u
#define TIMING_RUNS       10u
#define SAMPLE_RATE       10240u
#define WIN_LEN           1024
#define N_AXES            3
#define VREF              3.3f
#define GAIN              5.0f
#define GATE_THR_UNITS    0.01996f   /* 三轴联合 AC-RMS 门限（数据单位） */
#define INPUT_SCALE       0.047244094f
#define CLIP_I8           127.0f
#define ADC_MEAN_CODE     2048

static int clock_init_72mhz(void) {
    uint32_t timeout;
    RCC->CR |= RCC_CR_HSEON;
    timeout = 0x50000u;
    while (((RCC->CR & RCC_CR_HSERDY) == 0u) && (--timeout != 0u)) {}
    if ((RCC->CR & RCC_CR_HSERDY) == 0u) { SystemCoreClockUpdate(); return 0; }
    FLASH->ACR = FLASH_ACR_PRFTBE | FLASH_ACR_LATENCY_2;
    RCC->CFGR = (RCC->CFGR & ~(RCC_CFGR_HPRE | RCC_CFGR_PPRE1 | RCC_CFGR_PPRE2))
              | RCC_CFGR_HPRE_DIV1 | RCC_CFGR_PPRE1_DIV2 | RCC_CFGR_PPRE2_DIV1;
    RCC->CFGR = (RCC->CFGR & ~(RCC_CFGR_PLLSRC | RCC_CFGR_PLLXTPRE | RCC_CFGR_PLLMULL))
              | RCC_CFGR_PLLSRC | RCC_CFGR_PLLMULL9;
    RCC->CR |= RCC_CR_PLLON;
    timeout = 0x50000u;
    while (((RCC->CR & RCC_CR_PLLRDY) == 0u) && (--timeout != 0u)) {}
    if ((RCC->CR & RCC_CR_PLLRDY) == 0u) { SystemCoreClockUpdate(); return 0; }
    RCC->CFGR = (RCC->CFGR & ~RCC_CFGR_SW) | RCC_CFGR_SW_PLL;
    timeout = 0x50000u;
    while (((RCC->CFGR & RCC_CFGR_SWS) != RCC_CFGR_SWS_PLL) && (--timeout != 0u)) {}
    SystemCoreClockUpdate();
    return (RCC->CFGR & RCC_CFGR_SWS) == RCC_CFGR_SWS_PLL;
}

static uint32_t usart1_pclk2(void) {
    const uint32_t ppre2 = (RCC->CFGR & RCC_CFGR_PPRE2) >> RCC_CFGR_PPRE2_Pos;
    if (ppre2 < 4u) return SystemCoreClock;
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
        if (brr == 0u) brr = 1u;
        USART1->BRR = brr;
    }
    USART1->CR1 = USART_CR1_UE | USART_CR1_TE | USART_CR1_RE;
}

int _write(int file, char *ptr, int len) {
    (void)file;
    for (int i = 0; i < len; ++i) {
        while ((USART1->SR & USART_SR_TXE) == 0u) {}
        USART1->DR = (uint8_t)ptr[i];
    }
    return len;
}

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
    while ((uint32_t)(dwt_cycles() - start) < cycles) {}
}

static void gpio_analog_init(void) {
    RCC->APB2ENR |= RCC_APB2ENR_IOPAEN;
    GPIOA->CRL &= ~0x0000000Fu;
    GPIOA->CRL &= ~0x000F0000u;
}

static void adc_init(void) {
    RCC->APB2ENR |= RCC_APB2ENR_ADC1EN;
    RCC->CFGR = (RCC->CFGR & ~RCC_CFGR_ADCPRE) | RCC_CFGR_ADCPRE_DIV6;
    ADC1->CR1 = 0u;
    ADC1->CR2 = ADC_CR2_ADON;
    delay_cycles(24000u);
    {
        uint32_t to = 0x100000u;
        ADC1->CR2 |= ADC_CR2_RSTCAL;
        while ((ADC1->CR2 & ADC_CR2_RSTCAL) != 0u && (--to != 0u)) {}
        to = 0x100000u;
        ADC1->CR2 |= ADC_CR2_CAL;
        while ((ADC1->CR2 & ADC_CR2_CAL) != 0u && (--to != 0u)) {}
    }
    ADC1->SMPR1 = (ADC1->SMPR1 & ~0x7u) | 0x7u;
    ADC1->SQR1 = 0u;
    ADC1->SQR2 = 0u;
    ADC1->SQR3 = 0u;
    ADC1->CR2 |= ADC_CR2_EXTTRIG | (0x7u << 17) | ADC_CR2_CONT;
    ADC1->CR2 |= ADC_CR2_SWSTART;
    delay_cycles(24000u);
}

static void dac_init(void) {
    RCC->APB1ENR |= RCC_APB1ENR_DACEN;
    DAC->CR = DAC_CR_EN1;
    DAC->DHR12R1 = ADC_MEAN_CODE;
}

static uint16_t adc_read_once(void) {
    uint32_t to = 2000u;
    ADC1->CR2 |= ADC_CR2_SWSTART;
    while ((ADC1->SR & ADC_SR_EOC) == 0u && (--to != 0u)) {}
    if ((ADC1->SR & ADC_SR_EOC) == 0u) return ADC_MEAN_CODE;
    return (uint16_t)ADC1->DR;
}

static int16_t g_adc[N_AXES][WIN_LEN];
static int8_t g_net[MODEL_INPUT_C * MODEL_INPUT_LEN];
static float g_probs[MODEL_NUM_CLASSES];
static float g_last_infer_ms = 0.0f;

static uint32_t acquire_window(int cls) {
    const uint32_t t0 = dwt_cycles();
    uint32_t next = t0;
    const uint32_t step = SystemCoreClock / SAMPLE_RATE;
    for (int ax = 0; ax < N_AXES; ++ax) {
        for (int i = 0; i < WIN_LEN; ++i) {
            next += step;
            while ((int32_t)(dwt_cycles() - next) < 0) {}
            DAC->DHR12R1 = (uint16_t)REPLAY_XYZ[cls][ax][i];
            g_adc[ax][i] = (int16_t)adc_read_once();
        }
    }
    return dwt_cycles() - t0;
}

static const char *class_name(int c) {
    static const char *names[MODEL_NUM_CLASSES] = {"initial_wear", "steady_wear", "accelerating_wear"};
    if (c < 0 || c >= (int)MODEL_NUM_CLASSES) return "?";
    return names[c];
}

static void process_window(const int16_t adc[N_AXES][WIN_LEN], const char *tag) {
    int32_t sum = 0;
    int64_t sumsq = 0;
    const int n = N_AXES * WIN_LEN;
    for (int ax = 0; ax < N_AXES; ++ax) {
        for (int i = 0; i < WIN_LEN; ++i) {
            const int32_t v = (int32_t)adc[ax][i] - ADC_MEAN_CODE;
            sum += v;
            sumsq += (int64_t)v * (int64_t)v;
        }
    }
    const float mean_code = (float)sum / (float)n;
    const float rms_raw = sqrtf((float)((double)sumsq / (double)n));
    double var = 0.0;
    for (int ax = 0; ax < N_AXES; ++ax) {
        for (int i = 0; i < WIN_LEN; ++i) {
            const double d = (double)((int32_t)adc[ax][i] - ADC_MEAN_CODE) - (double)mean_code;
            var += d * d;
        }
    }
    const float rms_ac = sqrtf((float)(var / (double)n));
    const float rms_units = rms_ac * (VREF / 4095.0f) / GAIN;
    printf("[%s] mean=%.0f rms_ac=%.1fcode (%.3fV, %.4f unit) -> ",
           tag, (double)mean_code, (double)rms_ac,
           (double)(rms_ac * VREF / 4095.0f), (double)rms_units);
    if (rms_units < GATE_THR_UNITS) {
        printf("idling\r\n");
        return;
    }
    const float factor = 1.0f / (rms_raw * INPUT_SCALE);
    for (int ax = 0; ax < N_AXES; ++ax) {
        for (int i = 0; i < WIN_LEN; ++i) {
            float xn = (((float)adc[ax][i] - (float)ADC_MEAN_CODE) - mean_code) * factor;
            if (xn > CLIP_I8) xn = CLIP_I8;
            else if (xn < -CLIP_I8) xn = -CLIP_I8;
            int32_t q = (int32_t)(xn + (xn >= 0.0f ? 0.5f : -0.5f));
            if (q > 127) q = 127; else if (q < -128) q = -128;
            g_net[ax * WIN_LEN + i] = (int8_t)q;
        }
    }
    (void)model_forward(g_net, g_probs);
    uint32_t tsum = 0u;
    int pred = 0;
    for (uint32_t r = 0; r < TIMING_RUNS; ++r) {
        const uint32_t t1 = dwt_cycles();
        pred = (int)model_forward(g_net, g_probs);
        tsum += dwt_cycles() - t1;
    }
    g_last_infer_ms = (float)(tsum / TIMING_RUNS) * 1000.0f / (float)SystemCoreClock;
    printf("wear, pred=%d (%s) infer=%.2f ms\r\n", pred, class_name(pred), (double)g_last_infer_ms);
    for (int i = 0; i < (int)MODEL_NUM_CLASSES; ++i) {
        printf("    class %d (%s): %.4f\r\n", i, class_name(i), (double)g_probs[i]);
    }
    {
        float top1 = -1.0f, top2 = -1.0f;
        for (int i = 0; i < (int)MODEL_NUM_CLASSES; ++i) {
            if (g_probs[i] > top1) { top2 = top1; top1 = g_probs[i]; }
            else if (g_probs[i] > top2) { top2 = g_probs[i]; }
        }
        const float margin = top1 - top2;
        if (margin < 0.15f) {
            printf("[warn ] transition/uncertain: margin=%.3f (<0.15)，处于相邻阶段过渡区\r\n",
                   (double)margin);
        }
    }
}

int main(void) {
    (void)clock_init_72mhz();
    usart1_init();
    dwt_init();
    memcpy(_sramfunc, _siramfunc, (size_t)(_eramfunc - _sramfunc));
    gpio_analog_init();
    adc_init();
    dac_init();

    printf("\r\n[vibration-edge-xyz] 3-axis online ADC + int8 CNN\r\n");
    printf("build  : %s %s\r\n", __DATE__, __TIME__);
    printf("clock  : %u Hz, dwt=%s\r\n", (unsigned)SystemCoreClock, dwt_ok ? "ok" : "FAIL");
    printf("model  : %s, input %d x %d int8, classes %d, params %d\r\n",
           MODEL_NAME, (int)MODEL_INPUT_C, (int)MODEL_INPUT_LEN,
           (int)MODEL_NUM_CLASSES, (int)MODEL_PARAM_COUNT);
    printf("acq    : 3 axes x 1024 pts @ %u Hz (300 ms/window), DAC PA4 replay\r\n",
           (unsigned)SAMPLE_RATE);
    printf("gate   : joint AC-RMS < %.3f (data unit) => idling\r\n", (double)GATE_THR_UNITS);

    int cls = 0;
    for (;;) {
        printf("\r\n--- window: replay class %d (%s) idx=%d ---\r\n", cls, REPLAY_NAME[cls], REPLAY_IDX[cls]);
        const uint32_t acq = acquire_window(cls);
        printf("[acq  ] %u cycles (%.1f ms), 3 axes x 1024, adc_timeouts=0\r\n", (unsigned)acq,
               (double)((float)acq * 1000.0f / (float)SystemCoreClock));
        /* 供网页画波形：X 轴每 4 点取 1 点，256 点，约 1.4 KB 文本 */
        printf("[wave]");
        for (int i = 0; i < WIN_LEN; i += 4) {
            printf(" %d", (int)g_adc[0][i]);
        }
        printf("\r\n");
        process_window((const int16_t (*)[WIN_LEN])g_adc, "adc  ");
        process_window((const int16_t (*)[WIN_LEN])REPLAY_XYZ[cls], "replay");
        printf("[window] acq %.1f ms + infer %.2f ms = %.1f ms/decision (3-axis)\r\n",
               (double)((float)acq * 1000.0f / (float)SystemCoreClock),
               (double)g_last_infer_ms,
               (double)((float)acq * 1000.0f / (float)SystemCoreClock) + (double)g_last_infer_ms);
        cls = (cls + 1) % REPLAY_N;
        delay_cycles(SystemCoreClock * 3u / 2u);
    }
}
