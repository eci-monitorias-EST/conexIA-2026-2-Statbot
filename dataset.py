"""Carga y consultas sobre el dataset de partidos del Mundial 2026 (worldcup-full.json).

Expone funciones de solo lectura sobre los partidos, pensadas para ser usadas
como "tools" por el agente LLM en llm_agent.py: cada una calcula una respuesta
concreta (ranking, ficha de jugador, resumen) a partir de los datos crudos, en
vez de dejar que el modelo intente contar o sumar por su cuenta.
"""

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

DATASET_PATH = os.path.join(os.path.dirname(__file__), "worldcup-full.json")

with open(DATASET_PATH, encoding="utf-8") as _f:
    _data = json.load(_f)

TORNEO_NOMBRE: str = _data["name"]
PARTIDOS: List[Dict[str, Any]] = _data["matches"]


def _resultado_partido(p: Dict[str, Any]) -> Dict[str, Any]:
    """Marcador y ganador reales de un partido, considerando prórroga y penales.

    `score.ft` solo trae el resultado de los 90 minutos: en partidos de
    eliminatoria que se definen en prórroga (`score.et`) o penales
    (`score.p`) hay que mirar esas claves o el partido se ve como empate.
    """
    score = p["score"]
    goles = score.get("et", score["ft"])
    penales = score.get("p")
    if penales:
        decidido_en = "penales"
        ganador = p["team1"] if penales[0] > penales[1] else p["team2"]
    elif "et" in score:
        decidido_en = "prorroga"
        ganador = p["team1"] if goles[0] > goles[1] else (p["team2"] if goles[1] > goles[0] else None)
    else:
        decidido_en = "tiempo reglamentario"
        ganador = p["team1"] if goles[0] > goles[1] else (p["team2"] if goles[1] > goles[0] else None)
    return {
        "goles1": goles[0],
        "goles2": goles[1],
        "penales1": penales[0] if penales else None,
        "penales2": penales[1] if penales else None,
        "ganador": ganador,
        "decidido_en": decidido_en,
    }


def top_goleadores(n: int = 10, equipo: Optional[str] = None) -> List[Dict[str, Any]]:
    """Ranking de jugadores por goles anotados en el torneo, opcionalmente filtrado por selección."""
    n = int(n)
    goles = defaultdict(int)
    equipos: Dict[str, str] = {}
    for p in PARTIDOS:
        for lado, key in ((0, "goals1"), (1, "goals2")):
            eq = p["team1"] if lado == 0 else p["team2"]
            if equipo and equipo.lower() not in eq.lower():
                continue
            for g in p.get(key, []):
                nombre = g["name"]
                goles[nombre] += 1
                equipos[nombre] = eq
    ranking = sorted(goles.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return [{"jugador": nom, "equipo": equipos[nom], "goles": c} for nom, c in ranking]


def top_tarjetas(n: int = 10, tipo: Optional[str] = None) -> List[Dict[str, Any]]:
    """Ranking de jugadores con más tarjetas (amarillas y/o rojas) en el torneo."""
    n = int(n)
    conteo: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"amarillas": 0, "rojas": 0, "equipo": None})
    for p in PARTIDOS:
        equipos = (p["team1"], p["team2"])
        for lado, booking_list in enumerate(p.get("bookings", [])):
            for b in booking_list:
                nombre = b["name"]
                conteo[nombre]["equipo"] = equipos[lado]
                if b["type"] == "Y":
                    conteo[nombre]["amarillas"] += 1
                else:
                    conteo[nombre]["rojas"] += 1
    filas = [{"jugador": k, **v} for k, v in conteo.items()]
    if tipo == "roja":
        filas = [f for f in filas if f["rojas"] > 0]
        filas.sort(key=lambda f: f["rojas"], reverse=True)
    else:
        filas.sort(key=lambda f: (f["amarillas"] + f["rojas"]), reverse=True)
    return filas[:n]


def estadisticas_jugador(nombre: str) -> Dict[str, Any]:
    """Goles, tarjetas y partidos disputados por un jugador (búsqueda parcial por nombre)."""
    nombre_low = nombre.lower()
    goles: List[Dict[str, Any]] = []
    tarjetas: List[Dict[str, Any]] = []
    partidos: List[Dict[str, Any]] = []
    equipo: Optional[str] = None
    nombre_completo: Optional[str] = None

    for p in PARTIDOS:
        equipos = (p["team1"], p["team2"])
        rivales = (p["team2"], p["team1"])

        for lado, key in ((0, "goals1"), (1, "goals2")):
            for g in p.get(key, []):
                if nombre_low in g["name"].lower():
                    equipo, nombre_completo = equipos[lado], g["name"]
                    goles.append({
                        "rival": rivales[lado], "minuto": g["minute"],
                        "fecha": p["date"], "ronda": p["round"],
                    })

        bookings = p.get("bookings", [])
        for lado in (0, 1):
            if lado >= len(bookings):
                continue
            for b in bookings[lado]:
                if nombre_low in b["name"].lower():
                    equipo, nombre_completo = equipos[lado], b["name"]
                    tarjetas.append({
                        "tipo": "amarilla" if b["type"] == "Y" else "roja",
                        "rival": rivales[lado], "minuto": b["minute"],
                        "fecha": p["date"], "ronda": p["round"],
                    })

        lineup = p.get("lineup", [])
        for lado in (0, 1):
            if lado >= len(lineup):
                continue
            jugadores = lineup[lado].get("starter", []) + lineup[lado].get("bench", [])
            titulares = {j["name"] for j in lineup[lado].get("starter", [])}
            for j in jugadores:
                if nombre_low in j["name"].lower():
                    equipo, nombre_completo = equipos[lado], j["name"]
                    partidos.append({
                        "rival": rivales[lado], "fecha": p["date"], "ronda": p["round"],
                        "titular": j["name"] in titulares,
                    })
                    break

    if nombre_completo is None:
        return {"encontrado": False, "nombre_buscado": nombre}

    return {
        "encontrado": True,
        "nombre": nombre_completo,
        "equipo": equipo,
        "partidos_jugados": len(partidos),
        "goles": len(goles),
        "tarjetas_amarillas": sum(1 for t in tarjetas if t["tipo"] == "amarilla"),
        "tarjetas_rojas": sum(1 for t in tarjetas if t["tipo"] == "roja"),
        "detalle_goles": goles,
        "detalle_tarjetas": tarjetas,
    }


def partidos_de_equipo(equipo: str) -> List[Dict[str, Any]]:
    """Lista de partidos jugados por una selección, con resultado, ganador, sede y asistencia."""
    equipo_low = equipo.lower()
    resultado = []
    for p in PARTIDOS:
        es_local = equipo_low in p["team1"].lower()
        es_visita = equipo_low in p["team2"].lower()
        if not (es_local or es_visita):
            continue
        equipo_nombre = p["team1"] if es_local else p["team2"]
        rival = p["team2"] if es_local else p["team1"]
        r = _resultado_partido(p)
        goles_propios = r["goles1"] if es_local else r["goles2"]
        goles_rival = r["goles2"] if es_local else r["goles1"]
        resultado_texto = f"{goles_propios}-{goles_rival}"
        if r["decidido_en"] == "penales":
            pen_propios = r["penales1"] if es_local else r["penales2"]
            pen_rival = r["penales2"] if es_local else r["penales1"]
            resultado_texto += f" (penales {pen_propios}-{pen_rival})"
        resultado.append({
            "ronda": p["round"],
            "fecha": p["date"],
            "rival": rival,
            "resultado": resultado_texto,
            "gano": r["ganador"] == equipo_nombre,
            "empato": r["ganador"] is None,
            "decidido_en": r["decidido_en"],
            "sede": p["ground"],
            "asistencia": p.get("attendance"),
        })
    return resultado


def resumen_torneo() -> Dict[str, Any]:
    """Estadísticas generales del torneo: partidos, goles totales, promedios y asistencia."""
    total_partidos = len(PARTIDOS)
    total_goles = sum(len(p.get("goals1", [])) + len(p.get("goals2", [])) for p in PARTIDOS)
    asistencias = [p["attendance"] for p in PARTIDOS if p.get("attendance")]
    partido_mas_asistido = max(PARTIDOS, key=lambda p: p.get("attendance", 0))

    final = next((p for p in PARTIDOS if p["round"] == "Final"), None)
    campeon = subcampeon = resultado_final = None
    if final:
        r = _resultado_partido(final)
        campeon = r["ganador"]
        subcampeon = final["team2"] if campeon == final["team1"] else final["team1"]
        resultado_final = f"{final['team1']} {r['goles1']}-{r['goles2']} {final['team2']}"
        if r["decidido_en"] == "penales":
            resultado_final += f" (penales {r['penales1']}-{r['penales2']})"
        elif r["decidido_en"] == "prorroga":
            resultado_final += " (en prorroga)"

    return {
        "torneo": TORNEO_NOMBRE,
        "total_partidos": total_partidos,
        "total_goles": total_goles,
        "promedio_goles_por_partido": round(total_goles / total_partidos, 2) if total_partidos else None,
        "asistencia_promedio": round(sum(asistencias) / len(asistencias)) if asistencias else None,
        "asistencia_maxima": max(asistencias) if asistencias else None,
        "partido_mas_asistido": (
            f"{partido_mas_asistido['team1']} vs {partido_mas_asistido['team2']} "
            f"({partido_mas_asistido['ground']})"
        ),
        "campeon": campeon,
        "subcampeon": subcampeon,
        "resultado_final": resultado_final,
    }
