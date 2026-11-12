"""
common_legacy.py — camada de compatibilidade.
Os scripts antigos (figuras, Tabela 2, etc.) usavam funções próprias de preparo de dados
(prep_elsi, prep_pns, get_Xy, clean). Aqui elas passam a chamar o common.py, de modo que
TODAS as regras de codificação vivem em um único arquivo.
"""
from sklearn.model_selection import StratifiedKFold
from common import *                                    # noqa  (regras únicas)
from common import get_Xy as _get_Xy_new                # noqa

ELSI_W1 = DEFAULT_PATHS["elsi1"]; ELSI_W2 = DEFAULT_PATHS["elsi2"]
ELSI_W3 = DEFAULT_PATHS["elsi3"]; PNS13 = DEFAULT_PATHS["pns2013"]
PNS19 = DEFAULT_PATHS["pns2019"]
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def prep_elsi(path, wave):
    print(f"  Loading ELSI W{wave}...")
    return read_elsi(path, wave)


def prep_pns(path, sep, year):          # `sep` ignorado: definido no common
    print(f"  Loading PNS {year}...")
    return read_pns(path, year)


def get_Xy(df, target, age_min, age_max):
    """Assinatura antiga: devolve (X, y) sem pesos."""
    X, y, _ = _get_Xy_new(df, target, age_max, use_weights=False)
    return X, y
