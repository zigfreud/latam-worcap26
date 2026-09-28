"""Seleção interna e roteamento geográfico de especialistas congelados."""
import numpy as np

from .experiment import require


def route_brazil(base, specialist, mask, use_specialist):
    require(base.shape == specialist.shape and base.ndim == 3, "Esperado tempo/lat/lon iguais")
    require(base.dtype == specialist.dtype, "Tipos incompatíveis; preservar exterior exatamente")
    require(mask.dtype == np.bool_ and mask.shape == base.shape[1:], "Máscara incompatível")
    require(mask.any() and (~mask).any(), "Brasil e exterior devem existir")
    require(isinstance(use_specialist, (bool, np.bool_)), "Escolha deve ser booleana")
    require(np.isfinite(base).all() and np.isfinite(specialist).all(), "Previsão não finita")
    require((base >= 0).all() and (specialist >= 0).all(), "Previsão negativa")
    return np.where(mask[None] & use_specialist, specialist, base)


def choose_brazil(base, specialist, truth, mask):
    """Recebe apenas validação interna; empate fica com o controle."""
    routed = route_brazil(base, specialist, mask, True)
    require(truth.shape == base.shape and np.isfinite(truth).all(), "Alvo interno inválido")
    scores = []
    for selected, prediction in [(False, base), (True, routed)]:
        error = prediction[:, mask].astype(np.float64)-truth[:, mask]
        scores.append({"use_specialist": selected, "brazil_rmse": float(np.sqrt(np.square(error).mean())),
                       "count": int(error.size)})
    winner = min(scores, key=lambda row: row["brazil_rmse"])
    return winner["use_specialist"], scores
