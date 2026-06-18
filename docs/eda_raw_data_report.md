ANÁLISE EXPLORATÓRIA DOS DADOS BRUTOS
Gerado em: 2026-05-27 22:08:35
Script: scripts/eda_raw_data.py

========================================================================
  0. CONCATENAÇÃO DOS ARQUIVOS BRUTOS
========================================================================

Arquivo 1: METOFFICE-GLO-SST-L4-REP-OBS-SST_1755837049160(1).nc
  Período: 1996-05-31T00:00:00.000000000 a 2009-05-30T00:00:00.000000000
  Instantes: 4748

Arquivo 2: METOFFICE-GLO-SST-L4-REP-OBS-SST_1755836916104(1).nc
  Período: 2009-05-31T00:00:00.000000000 a 2022-05-31T00:00:00.000000000
  Instantes: 4749

Grades espaciais idênticas: lat=True, lon=True

Após concatenação: 9497 instantes
Instantes duplicados encontrados: 0
Série final ordenada: 1996-05-31T00:00:00.000000000 a 2022-05-31T00:00:00.000000000

========================================================================
  1. ESTRUTURA DO DATASET CONCATENADO
========================================================================

Dimensões: {'time': 9497, 'latitude': 100, 'longitude': 100}
Variáveis: ['analysed_sst', 'analysis_error', 'mask', 'sea_ice_fraction']
Coordenadas: ['time', 'latitude', 'longitude']

  analysed_sst: dtype=float32, shape=(9497, 100, 100), 379.9 MB
  analysis_error: dtype=float32, shape=(9497, 100, 100), 379.9 MB
  mask: dtype=float32, shape=(9497, 100, 100), 379.9 MB
  sea_ice_fraction: dtype=float32, shape=(9497, 100, 100), 379.9 MB

Memória total (arrays): 1519.5 MB

Atributos globais relevantes:
  title: Global SST & Sea Ice Analysis, L4 OSTIA, 0.05 deg daily (METOFFICE-GLO-SST-L4-REP-OBS-SST-V2)
  institution: UKMO
  source: AMSR2-REMSS-L2P-v2.0, AMSRE-REMSS-L2P-v2.0, GOES<13,16>-OSISAF-L3C-v2.0, SEVIRI-OSISAF-L3C-v2.0, SLSTRA-C3S-L3C-v2.0, ATSR<1,2>-ESACCI-L3U-v2.0, AATSR-ESACCI-L3U-v2.0, AVHRR<07,09,11,12,14,15,16,17,18,19,MTA>-ESACCI-L3U-v2.0, GMI-REMSS-L3U-v2.0, VIIRS<NPP,N20>-OSPO-L3U-v2.0, OSISAF_ICE
  Conventions: CF-1.11

========================================================================
  2. COORDENADAS
========================================================================

TEMPO:
  Primeiro instante: 1996-05-31T00:00:00.000000000
  Último instante:   2022-05-31T00:00:00.000000000
  Total de instantes (T): 9497
  Diferença mín entre instantes: 1 dia(s)
  Diferença máx entre instantes: 1 dia(s)
  Diferença média:               1.000000 dia(s)
  Lacunas temporais: NENHUMA (série diária contínua)

LATITUDE:
  Min: -29.98°   Max: -25.02°
  Pontos (H): 100
  Espaçamento: 0.0500° (min=0.0500°, max=0.0500°)

LONGITUDE:
  Min: -48.98°   Max: -44.03°
  Pontos (W): 100
  Espaçamento: 0.0500° (min=0.0500°, max=0.0500°)

GRADE TOTAL: 100 x 100 = 10000 pixels
Resolução nominal: 0.05° ≈ 5.6 km

========================================================================
  3. ANALYSED_SST — ESTATÍSTICAS GLOBAIS
========================================================================

Shape: (T=9497, H=100, W=100)
dtype: float32
Total de elementos: 94,970,000

CONTAGEM:
  Elementos válidos (finitos): 89,699,165 (94.4500%)
  Elementos NaN:               5,270,835 (5.5500%)

ESTATÍSTICAS (Kelvin):
  Mínimo:          285.3200 K
  Máximo:          304.8600 K
  Amplitude:       19.5400 K
  Média:           296.3627 K
  Mediana:         296.2800 K
  Desvio padrão:   2.4936 K
  Variância:       6.2178 K²

REFERÊNCIA EM CELSIUS (K − 273.15, sem alterar o array):
  Mínimo:          12.1700 °C
  Máximo:          31.7100 °C
  Amplitude:       19.5400 °C
  Média:           23.2127 °C
  Mediana:         23.1300 °C
  Desvio padrão:   2.4936 °C


========================================================================
  4. ANALYSED_SST — VARIABILIDADE ESPACIAL (POR PIXEL)
========================================================================

Cálculo: média, desvio, mín e máx temporais por pixel (eixo T=9497)
  np.nanmean(sst, axis=0)  → média ignorando NaN
  np.nanstd(sst, axis=0, ddof=0)  → desvio padrão populacional
  np.nanmin / np.nanmax  → extremos por pixel

CLASSIFICAÇÃO DE PIXELS:
  Oceano (pelo menos 1 valor finito): 9445 pixels (94.45%)
  Terra/gelo (todos NaN):             555 pixels (5.55%)

MÉDIA TEMPORAL POR PIXEL (apenas oceano, Kelvin / °C):
  Min das médias:   293.7514 K  (20.6014 °C)
  Max das médias:   297.7847 K  (24.6347 °C)
  Média das médias: 296.3620 K  (23.2120 °C)

DESVIO PADRÃO TEMPORAL POR PIXEL (apenas oceano):
  Min dos desvios:   1.7887 K
  Max dos desvios:   3.2333 K
  Média dos desvios: 2.3497 K
  Pixels com σ > 1.5: 9445 (100.0%)
  Pixels com σ > 2.0: 8124 (86.0%)
  Pixels com σ > 2.5: 2889 (30.6%)
  Pixels com σ > 3.0: 1455 (15.4%)

AMPLITUDE TÉRMICA POR PIXEL (max − min temporal, apenas oceano):
  Min amplitude:   7.8800 K
  Max amplitude:   16.8800 K
  Média amplitude: 11.6935 K

========================================================================
  5. ESTRUTURA DE NaN
========================================================================

POR PIXEL (ao longo de T=9497 instantes):
  Sempre NaN (terra):              555 pixels
  Nunca NaN (oceano sem falhas):    9445 pixels
  Parcialmente NaN (falhas pontuais): 0 pixels

POR INSTANTE TEMPORAL:
  NaN por frame — min: 555, max: 555
  NaN por frame — média: 555.00, desvio: 0.0000

  Linha de base (NaN fixos = terra): 555 pixels/frame
  Instantes com NaN acima da linha de base: 0 de 9497 (0.00%)

FRAÇÃO DE NaN TOTAL:
  5,270,835 de 94,970,000 elementos (5.5500%)
  NaN atribuíveis a terra (baseline × T): 5,270,835 (5.5500%)
  NaN em pixels oceânicos:                0 (0.000000%)

========================================================================
  6. MASK (BITMASK)
========================================================================

Shape: (T=9497, H=100, W=100)
dtype: float32

Valores únicos em toda a série: [1. 2.]

DISTRIBUIÇÃO GLOBAL:
  valor=1: 89,699,165 ocorrências (94.45%)
  valor=2: 5,270,835 ocorrências (5.55%)

DISTRIBUIÇÃO NO PRIMEIRO INSTANTE (t=0):
  valor=1: 9445 pixels (94.45%)
  valor=2: 555 pixels (5.55%)

VARIAÇÃO TEMPORAL:
  Mask é CONSTANTE no tempo (50 instantes amostrados, passo=189)

DECOMPOSIÇÃO POR BIT (OSTIA bitmask, primeiro instante):
  bit 0 — água aberta: 9445 pixels (94.45%)
  bit 1 — terra: 555 pixels (5.55%)
  bit 2 — lago: 0 pixels (0.00%)
  bit 3 — gelo marinho: 0 pixels (0.00%)

========================================================================
  7. SEA_ICE_FRACTION
========================================================================

Shape: (T=9497, H=100, W=100)
dtype: float32

NaN: 5,270,835 (5.5500%)
Válidos: 89,699,165
  Valores únicos: [0.]
  Min: 0.000000
  Max: 0.000000
  Média: 0.000000
  Valores > 0: 0 (0.000000%)

  CONCLUSÃO: sea_ice_fraction é ZERO em todo o domínio e período.
  Região subtropical sem gelo marinho.

========================================================================
  8. ANALYSIS_ERROR
========================================================================

Shape: (T=9497, H=100, W=100)
dtype: float32

NaN: 5,270,835 (5.5500%)
Válidos: 89,699,165

ESTATÍSTICAS:
  Min:    0.210000
  Max:    1.420000
  Média:  0.363349
  Mediana: 0.300000
  Desvio: 0.182057


========================================================================
Sintese
========================================================================

Período:             1996-05-31T00:00:00.000000000 a 2022-05-31T00:00:00.000000000
Instantes (T):       9497
Lacunas temporais:   nenhuma
Grade:               100 × 100 = 10000 pixels
Resolução:           0.05° ≈ 5.6 km
Latitude:            -29.98° a -25.02°
Longitude:           -48.98° a -44.03°
Pixels oceânicos:    9445 (94.45%)
Pixels terra:        555 (5.55%)

Unidade da SST:      Kelvin (sem conversão)
SST mín:             285.3200 K (12.1700 °C)
SST máx:             304.8600 K (31.7100 °C)
SST média:           296.3627 K (23.2127 °C)
SST desvio padrão:   2.4936 K

NaN total:           5,270,835 de 94,970,000 (5.5500%)
NaN por frame:       constante = 555 (min=max=True)
NaN oceânicos:       0
sea_ice_fraction:    toda zero
Variáveis:           analysed_sst, analysis_error, mask, sea_ice_fraction