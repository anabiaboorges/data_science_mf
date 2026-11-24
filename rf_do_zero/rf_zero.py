"""RF do zero – funções usadas pelos notebooks passo_XX.

Mantém do projeto original apenas a separação do tempo:
  treino     : tudo antes de 2024 (janela deslizante – ver passo 4)
  validação  : 2024                  -> onde fazemos TODAS as escolhas
  teste      : 01/2025 a 03/2026     -> rodado uma única vez, no final
  extra      : 04/2026 a 09/2026     -> (só WIN) nunca usado em nenhuma escolha
"""
import numpy as np
import pandas as pd

VALIDACAO = ("2024-01-01", "2025-01-01")
TESTE = ("2025-01-01", "2026-04-01")
EXTRA = ("2026-04-01", "2026-10-01")

CUSTO_PTS = 5          # custo ida+volta por contrato, em pontos (corretagem + emolumentos + slippage)
REAIS_POR_PONTO = {"WIN": 0.20, "WDO": 10.00}

# Pregão regular (fora disso o mercado é ralo e o candle não é comparável)
INICIO_PREGAO, FIM_PREGAO = "09:00", "18:00"


def carregar(caminho):
    """Lê o CSV do MetaTrader (separador TAB, cabeçalho entre < >)."""
    df = pd.read_csv(caminho, sep="\t")
    df.columns = [c.strip("<>").lower() for c in df.columns]
    df.index = pd.to_datetime(df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M:%S")
    df = df.drop(columns=["date", "time", "spread"])
    return df.between_time(INICIO_PREGAO, FIM_PREGAO, inclusive="left")


def reamostrar(df_1min, minutos):
    """Agrupa candles de 1 min em candles de N min, sem misturar dias."""
    if minutos == 1:
        return df_1min.copy()
    agg = {"open": "first", "high": "max", "low": "min", "close": "last",
           "tickvol": "sum", "vol": "sum"}
    return df_1min.resample(f"{minutos}min", origin="start_day").agg(agg).dropna()


def adicionar_alvo(df):
    """Alvo em PONTOS: quanto o preço anda do fechamento de t até o fechamento de t+1.

    É exatamente o resultado (antes do custo) de comprar 1 contrato no fechamento do candle t
    e zerar no fechamento do candle t+1. O último candle do dia fica sem alvo: não levamos
    posição de um dia para o outro.
    """
    df = df.copy()
    dia = df.index.normalize()
    df["mov_prox"] = df["close"].shift(-1) - df["close"]
    ultimo_do_dia = dia != np.roll(dia, -1)
    ultimo_do_dia[-1] = True
    df.loc[ultimo_do_dia, "mov_prox"] = np.nan
    # mesmo movimento em % (log-retorno x 100): não depende do nível do preço
    df["ret_prox"] = np.log(df["close"].shift(-1) / df["close"]) * 100
    df.loc[ultimo_do_dia, "ret_prox"] = np.nan
    return df


def periodo(df, intervalo):
    ini, fim = intervalo
    return df[(df.index >= ini) & (df.index < fim)]


def resultado_operacoes(direcao, mov_prox, custo=CUSTO_PTS):
    """Régua de dinheiro, usada em TODOS os passos.

    direcao: +1 compra, -1 vende, 0 fica de fora (um valor por candle)
    mov_prox: movimento do próximo candle, em pontos
    Devolve o resultado em pontos de cada operação feita (já descontado o custo).
    """
    direcao = np.asarray(direcao)
    mov = np.asarray(mov_prox)
    opera = direcao != 0
    return direcao[opera] * mov[opera] - custo


# ---------------------------------------------------------------- Passo 3: variáveis
JANELA_ROLL = 20   # nº de candles usados nas médias/desvios móveis


def criar_variaveis(df, n_lags=10):
    """Variáveis conhecidas no FECHAMENTO do candle t (nada do futuro).

    Precisa de um df já com o alvo (adicionar_alvo). Devolve df + colunas novas e o
    dicionário {grupo: [colunas]}.
    """
    df = df.copy()
    dia = df.index.normalize()
    novo_dia = pd.Series(dia, index=df.index).diff().ne(pd.Timedelta(0)).values
    # retorno do candle t (%); no 1º candle do dia usa abertura->fechamento (sem o gap da noite)
    ret = np.log(df["close"] / df["close"].shift(1)) * 100
    ret[novo_dia] = (np.log(df["close"] / df["open"]) * 100)[novo_dia]
    grupos = {}

    grupos["lags"] = []
    for k in range(1, n_lags + 1):
        df[f"ret_lag{k}"] = ret.shift(k - 1)          # lag1 = retorno do próprio candle t
        grupos["lags"].append(f"ret_lag{k}")

    amp = df["high"] - df["low"]
    df["amplitude"] = amp / df["close"] * 100
    df["corpo"] = (df["close"] - df["open"]) / df["close"] * 100
    df["pos_fech"] = ((df["close"] - df["low"]) / amp.replace(0, np.nan)).fillna(0.5)
    grupos["forma"] = ["amplitude", "corpo", "pos_fech"]

    df["volat"] = ret.rolling(JANELA_ROLL).std()
    df["volat_rel"] = df["volat"] / df["volat"].rolling(5 * JANELA_ROLL).mean()
    grupos["volatilidade"] = ["volat", "volat_rel"]

    df["vol_rel"] = df["vol"] / df["vol"].rolling(JANELA_ROLL).mean()
    grupos["volume"] = ["vol_rel"]

    g = df.groupby(dia)
    abertura_dia = g["open"].transform("first")
    df["hora"] = df.index.hour + df.index.minute / 60
    df["ret_dia"] = np.log(df["close"] / abertura_dia) * 100
    df["dist_max_dia"] = (df["close"] / g["high"].cummax() - 1) * 100
    df["dist_min_dia"] = (df["close"] / g["low"].cummin() - 1) * 100
    grupos["contexto_dia"] = ["hora", "ret_dia", "dist_max_dia", "dist_min_dia"]

    df = df.replace([np.inf, -np.inf], np.nan)
    return df, grupos


def avaliar(prev_pct, base, frac_fortes=0.2):
    """Métricas de previsão SEM custo (o custo só entra na regra de operação)."""
    real = base["ret_prox"].values
    mov = base["mov_prox"].values
    prev_pct = np.asarray(prev_pct)
    eqm, eqm_naive = np.mean((real - prev_pct) ** 2), np.mean(real ** 2)
    direcao = np.sign(prev_pct)
    bruto = direcao * mov
    fortes = np.abs(prev_pct) >= np.quantile(np.abs(prev_pct), 1 - frac_fortes)
    nz = (direcao != 0) & (real != 0)
    return pd.Series({
        "R² vs naive %": 100 * (1 - eqm / eqm_naive),
        "Theil U": np.sqrt(eqm / eqm_naive),           # = raiz(1 − R²); U < 1 ⇔ R² > 0
        "acerto direção %": 100 * np.mean(direcao[nz] == np.sign(real[nz])),
        "ganho bruto pts/op": bruto.mean(),
        "t ganho bruto": bruto.mean() / (bruto.std(ddof=1) / np.sqrt(len(bruto))),
        f"ganho bruto {int(frac_fortes*100)}% fortes": bruto[fortes].mean(),
    })


# ---------------------------------------------------------------- Passo 4: walk-forward
from sklearn.ensemble import RandomForestRegressor

N_LAGS = 10  # decidido no Passo 3 (Fase A, maior R² médio); versão 1 usava 5


def colunas(grupos, n_lags=N_LAGS):
    lags = [f"ret_lag{i}" for i in range(1, n_lags + 1)]
    return lags + [c for g, cs in grupos.items() if g != "lags" for c in cs]


def rf_padrao(n_linhas, **kw):
    """RF moderada usada nos Passos 3–5 (hiperparâmetros só são ajustados no Passo 6)."""
    p = dict(n_estimators=300, max_depth=6, min_samples_leaf=max(20, int(0.005 * n_linhas)),
             max_features=0.5, n_jobs=-1, random_state=42)
    p.update(kw)
    return RandomForestRegressor(**p)


def montar_base(df_1min, minutos):
    """Candles de N min + alvo + variáveis, sem linhas incompletas."""
    b, grupos = criar_variaveis(adicionar_alvo(reamostrar(df_1min, minutos)), N_LAGS)
    cols = colunas(grupos)
    return b.dropna(subset=cols + ["ret_prox"]), cols


def walk_forward(base, cols, intervalo, janela_meses, freq="QS", fabrica=rf_padrao):
    """Re-treina no início de cada bloco (MS=mensal, QS=trimestral, 6MS=semestral) com os
    últimos `janela_meses` meses e prevê o bloco seguinte. Devolve a base do intervalo + 'prev'."""
    ini, fim = pd.Timestamp(intervalo[0]), pd.Timestamp(intervalo[1])
    inicios = list(pd.date_range(ini, fim, freq=freq, inclusive="left"))
    if not inicios or inicios[0] != ini:
        inicios = [ini] + inicios
    saidas = []
    for a, b in zip(inicios, inicios[1:] + [fim]):
        tr = base[(base.index >= a - pd.DateOffset(months=janela_meses)) & (base.index < a)]
        te = base[(base.index >= a) & (base.index < b)]
        if len(te) == 0:
            continue
        rf = fabrica(len(tr)).fit(tr[cols], tr["ret_prox"])
        saidas.append(te.assign(prev=rf.predict(te[cols])))
    return pd.concat(saidas)


def resumo_wf(prev_df, custo=CUSTO_PTS, frac_fortes=0.2):
    """Resumo de um walk-forward: sinal (sem custo) e dinheiro (com custo, operando sempre)."""
    m = avaliar(prev_df["prev"].values, prev_df, frac_fortes)
    bruto = np.sign(prev_df["prev"]) * prev_df["mov_prox"]
    tri = bruto.groupby(prev_df.index.quarter).mean()
    m["trimestres + (bruto)"] = int((tri > 0).sum())
    m["líquido pts/op (sempre)"] = m["ganho bruto pts/op"] - custo
    m[f"líquido {int(frac_fortes*100)}% fortes"] = m[f"ganho bruto {int(frac_fortes*100)}% fortes"] - custo
    m["n_ops"] = len(prev_df)
    return m


# ---------------------------------------------------------------- Passo 5: regra de operação
def prev_pontos(d):
    """Converte a previsão (% de retorno) em pontos: % / 100 x preço de fechamento."""
    return d["prev"] / 100 * d["close"]


def resumo_ops(res_pts, datas, reais_por_ponto=REAIS_POR_PONTO["WIN"]):
    """Resumo de operações já com custo: esperança, t, trimestres, total em R$, drawdown."""
    res = pd.Series(np.asarray(res_pts, dtype=float), index=datas)
    n = len(res)
    if n < 2:
        return pd.Series({"n_ops": n})
    capital = res.cumsum()
    tri = res.groupby(res.index.quarter).mean()
    return pd.Series({
        "n_ops": n,
        "esperança pts/op": res.mean(),
        "t": res.mean() / (res.std(ddof=1) / np.sqrt(n)),
        "acerto %": 100 * (res > 0).mean(),
        "trimestres +": int((tri > 0).sum()),
        "total pts": res.sum(),
        "total R$ (1 contrato)": res.sum() * reais_por_ponto,
        "drawdown máx pts": (capital.cummax() - capital).max(),
    })


# ---------------------------------------------------------------- Fase A: só acerto (R²), sem custo
def r2_oos(prev_pct, real_pct):
    """R² fora da amostra contra prever zero, em %. Critério único de escolha da Fase A."""
    real, prev = np.asarray(real_pct), np.asarray(prev_pct)
    return 100 * (1 - np.mean((real - prev) ** 2) / np.mean(real ** 2))


def resumo_acerto(d):
    """Medidas de acerto de um walk-forward (df com 'prev' e 'ret_prox'). Nenhuma usa custo ou pontos."""
    real, prev = d["ret_prox"].values, d["prev"].values
    nz = (real != 0) & (prev != 0)
    tri = pd.Series({q: r2_oos(g["prev"], g["ret_prox"]) for q, g in d.groupby(d.index.quarter)})
    r2 = r2_oos(prev, real)
    return pd.Series({"R² %": r2, "Theil U": np.sqrt(1 - r2 / 100),
                      "acerto direção %": 100 * np.mean(np.sign(prev[nz]) == np.sign(real[nz])),
                      "corr prev×real": np.corrcoef(prev, real)[0, 1],
                      "trimestres R²>0": int((tri > 0).sum()), "n": len(d)})


def fabrica_congelada(cfg):
    """RF com os hiperparâmetros escolhidos na Fase A (lidos de saidas/modelo_congelado.json)."""
    return lambda n: RandomForestRegressor(n_estimators=300, max_depth=cfg["max_depth"],
                                           min_samples_leaf=cfg["min_samples_leaf"],
                                           max_features=cfg["max_features"], n_jobs=-1, random_state=42)


FREQ_CODIGO = {"mensal": "MS", "trimestral": "QS", "semestral": "6MS"}
