"""
modelo_base.py — Modelo 1: selección de polígonos, compras y edad en bodega.

Lo importan los dos notebooks del método ε-constraint (AUGMECON2):
    barrido_maxCarbono.ipynb   ->  max carbono   s.a. costo   <= ε
    barrido_minCosto.ipynb     ->  min costo     s.a. carbono >= ε

Flujo:
    1. cargar_datos()            lee adyacencia, supervivencia y captura (CSV)
    2. construir_modelo(datos)   arma el MIP completo (sin resolver)
    3. obtener_anclas(...)       tabla de pagos lexicográfica (se calcula una vez y se guarda)
    4. barrido(..., direccion)   barre ε sobre el rango de la tabla de pagos (AUGMECON2)
    5. combinar_fronteras()      une ambos barridos y quita puntos dominados

Todo lo editable está en `cfg` (ver clase Config). Desde el notebook:
    import modelo_base as mb
    mb.cfg.N_PUNTOS = 10
    mb.cfg.LIBERAR_SELECCION = True
"""
import os
import math
import time
import pickle
import hashlib
from dataclasses import dataclass, asdict

import pandas as pd
import pyomo.environ as pyo


# =====================================================================================
# Datos fijos del caso
# =====================================================================================
ESPECIES = ['Agave lechuguilla', 'Agave salmiana', 'Agave scabra', 'Agave striata',
            'Opuntia cantabrigiensis', 'Opuntia engelmani', 'Opuntia robusta',
            'Opuntia streptacanta', 'Prosopis laevigata', 'Yucca filifera']

# plantas por hectárea (columna HA del Cuadro 1); suman 524
DENS = {'Agave lechuguilla': 33, 'Agave salmiana': 157, 'Agave scabra': 33,
        'Agave striata': 33, 'Opuntia cantabrigiensis': 39, 'Opuntia engelmani': 30,
        'Opuntia robusta': 58, 'Opuntia streptacanta': 51, 'Prosopis laevigata': 69,
        'Yucca filifera': 21}

# hectáreas por polígono (transcritas del mapa; verifica)
HA_TODOS = {
    1: 5.4,   2: 7.52,  3: 8.0,   4: 8.0,   5: 7.56,  6: 4.19,  7: 6.28,  8: 7.6,
    9: 8.0,   10: 8.0,  11: 7.67, 12: 1.47, 13: 7.97, 14: 5.98, 15: 5.4,  16: 5.64,
    17: 6.11, 18: 7.11, 19: 4.92, 20: 1.38, 21: 8.0,  22: 7.82, 23: 5.53, 24: 5.64,
    25: 5.05, 26: 4.75, 27: 1.28, 28: 6.64, 29: 6.54, 30: 6.76, 31: 7.34,
}

VIVEROS = ['Vivero', 'Moctezuma', 'Venado', 'Laguna seca']

COSTOS_ESPECIES = {   # costo de compra de la especie s en el vivero j
    ('Agave lechuguilla', 'Laguna seca'): 26,
    ('Agave salmiana', 'Laguna seca'): 26,
    ('Agave scabra', 'Laguna seca'): 26,
    ('Agave scabra', 'Moctezuma'): 26,
    ('Agave striata', 'Moctezuma'): 26,
    ('Agave striata', 'Venado'): 25,
    ('Opuntia cantabrigiensis', 'Moctezuma'): 17,
    ('Opuntia cantabrigiensis', 'Venado'): 18,
    ('Opuntia engelmani', 'Venado'): 18,
    ('Opuntia engelmani', 'Laguna seca'): 21,
    ('Opuntia robusta', 'Moctezuma'): 17,
    ('Opuntia robusta', 'Venado'): 18,
    ('Opuntia robusta', 'Laguna seca'): 18,
    ('Opuntia streptacanta', 'Venado'): 18,
    ('Prosopis laevigata', 'Vivero'): 26.5,
    ('Yucca filifera', 'Vivero'): 26,
}

OFERTA_ESPECIES = {   # oferta por día de la especie s en el vivero j
    ('Agave lechuguilla', 'Laguna seca'): 500,
    ('Agave salmiana', 'Laguna seca'): 1500,
    ('Agave scabra', 'Laguna seca'): 500,
    ('Agave scabra', 'Moctezuma'): 500,
    ('Agave striata', 'Moctezuma'): 500,
    ('Agave striata', 'Venado'): 500,
    ('Opuntia cantabrigiensis', 'Moctezuma'): 250,
    ('Opuntia cantabrigiensis', 'Venado'): 250,
    ('Opuntia engelmani', 'Venado'): 180,
    ('Opuntia engelmani', 'Laguna seca'): 210,
    ('Opuntia robusta', 'Moctezuma'): 300,
    ('Opuntia robusta', 'Venado'): 300,
    ('Opuntia robusta', 'Laguna seca'): 70,
    ('Opuntia streptacanta', 'Venado'): 500,
    ('Prosopis laevigata', 'Vivero'): 700,
    ('Yucca filifera', 'Vivero'): 500,
}


# =====================================================================================
# Configuración editable
# =====================================================================================
@dataclass
class Config:
    # --- rutas ---
    CARPETA: str = os.path.dirname(os.path.abspath(__file__))   # donde viven los CSV de entrada y salen los resultados

    # --- horizonte, bodega y siembra ---
    EDADES_PLANTABLES: tuple = (3, 4, 5, 6, 7)   # la planta llega con edad 0; se planta con 3 a 7 días en bodega
    N_DIAS: int = 30                             # horizonte; inventario inicial = 0
    BANDA: float = 0.10                          # tolerancia de +-10% sobre la densidad por hectárea
    MIN_POR_PLANTA: int = 5
    J_LAB: int = 360                             # minutos por jornada
    TRABAJADORES: int = 30                       # por día (fijo)
    HA_MIN_DIA: float = 4.0                      # ha mínimas por día desde el primer día plantable
    HA_LOTE_MIN: float = 0.25                    # si un polígono se trabaja un día, al menos esta superficie

    # --- costos y capacidades ---
    K_CAM: int = 8000          # capacidad del camión (plantas por viaje)
    C_CAM: float = 4500        # costo por viaje
    C_PLANTAR: float = 20      # costo de plantar una planta
    K_ALM: float = 400         # m2 de bodega
    AREA_S: float = 0.04       # m2 por planta
    C_MERMA: float = 20        # costo de descartar una planta

    # --- método ---
    # MODO DE LA CORRIDA: se cambia SOLO aquí, para que ambos notebooks usen siempre el mismo.
    # False -> selFija (se fija el calendario del plan de costo mínimo)   | resultados_selFija/
    # True  -> selLibre (fase 2 con selección libre; lenta, puede atorarse) | resultados_selLibre/
    LIBERAR_SELECCION: bool = False
    N_PUNTOS: int = 10               # intervalos de la malla de ε (N_PUNTOS + 1 puntos contando las anclas)
    RHO: float = 1e-3                # peso del término aumentado de AUGMECON
    USAR_BYPASS: bool = True         # salto de AUGMECON2 sobre ε redundantes
    REANUDAR: bool = True            # reutiliza los puntos ya resueltos (barrido_<dir>/<dir>_pXX.pkl) si se interrumpió
    REINTENTOS: int = 3
    FACTOR_TIEMPO: float = 2.0       # el factor que multiplica al tiempo si no se llega al optimal en el tiempo 
    # --- decisión a posteriori sobre la frontera ---
    PESO_CARBONO: float = 0.5        # peso w del carbono en la suma ponderada normalizada (1 - w para el costo)
    PRECIO_CARBONO: float = None     # $ por unidad de carbono; si se da, se elige además el punto con max p·carbono - costo

    # --- solver ---
    SOLVER: str = 'appsi_highs'
    TIEMPO_FIJA: int = 120           # fase 1 de cada resolución: selección de polígonos fija (rápida)
    TIEMPO_PUNTO: int = 300          # fase 2 (solo selLibre): selección libre arrancando de la fase 1
    TIEMPO_ETAPA0: int = 300         # segundos para el calendario factible inicial
    GAP: float = 0.01                # brecha relativa aceptada

    # ---------- derivados (no editar) ----------
    @property
    def EDAD_MIN(self): return min(self.EDADES_PLANTABLES)
    @property
    def EDAD_MAX(self): return max(self.EDADES_PLANTABLES)
    @property
    def DIAS(self): return list(range(1, self.N_DIAS + 1))
    @property
    def PRIMER_DIA_PLANTABLE(self): return 1 + self.EDAD_MIN            # llega el día 1 -> se planta desde el día 4
    @property
    def ULTIMO_DIA_LLEGADA(self): return self.N_DIAS - self.EDAD_MIN     # lo que llega después no alcanza edad 3
    @property
    def CAP_TRAB(self): return self.J_LAB // self.MIN_POR_PLANTA        # plantas por trabajador por día
    @property
    def DENS_TOTAL(self): return sum(DENS.values())                     # 524 plantas por hectárea
    @property
    def MODO(self): return 'selLibre' if self.LIBERAR_SELECCION else 'selFija'
    @property
    def CARPETA_RESULTADOS(self): return os.path.join(self.CARPETA, f'resultados_{self.MODO}')


cfg = Config()

# parámetros que NO cambian la tabla de pagos (no invalidan las anclas guardadas)
_NO_AFECTAN_ANCLAS = {'CARPETA', 'N_PUNTOS', 'RHO', 'USAR_BYPASS', 'REANUDAR', 'SOLVER', 'TIEMPO_FIJA', 'TIEMPO_PUNTO',
                      'TIEMPO_ETAPA0', 'GAP', 'PESO_CARBONO', 'PRECIO_CARBONO'}
COMPONENTES_COSTO = ('compra', 'camion', 'siembra', 'merma')
VERSION_ANCLAS = 2   # cambia cuando cambia el procedimiento de las anclas: invalida las guardadas


# =====================================================================================
# Lectura de datos externos (CSV)
# =====================================================================================
def _ruta(nombre):
    return os.path.join(cfg.CARPETA, nombre)


def cargar_datos():
    """Lee adyacencia.csv, supervivencia.csv y captura.csv de cfg.CARPETA."""
    # --- adyacencia ---
    df = pd.read_csv(_ruta('adyacencia.csv'), encoding='utf-8-sig')
    pares = {frozenset((int(a), int(b))) for a, b in zip(df.poligono_a, df.poligono_b) if int(a) != int(b)}
    desconocidos = {h for p in pares for h in p if h not in HA_TODOS}
    assert not desconocidos, f'Polígonos inexistentes en la adyacencia: {desconocidos}'
    print(f'Adyacencia: {len(df)} filas en el CSV -> {len(pares)} pares únicos')
    if 'fuente' in df:
        print('  fuente:', '; '.join(df['fuente'].dropna().unique()))

    # --- supervivencia por especie y edad ---
    df = pd.read_csv(_ruta('supervivencia.csv'), encoding='utf-8-sig')
    pi = {(r.especie, int(r.edad)): float(r.supervivencia) for r in df.itertuples()}
    faltan = [(s, a) for s in ESPECIES for a in cfg.EDADES_PLANTABLES if (s, a) not in pi]
    assert not faltan, f'Faltan combinaciones especie/edad en supervivencia.csv: {faltan[:5]}...'
    if 'fuente' in df:
        print('Supervivencia:', '; '.join(df['fuente'].dropna().unique()))

    # --- captura por planta sobreviviente ---
    df = pd.read_csv(_ruta('captura.csv'), encoding='utf-8-sig')
    cap = {r.especie: float(r.captura_por_planta) for r in df.itertuples()}
    faltan = [s for s in ESPECIES if s not in cap]
    assert not faltan, f'Faltan especies en captura.csv: {faltan}'
    if 'fuente' in df:
        print('Captura:', '; '.join(df['fuente'].dropna().unique()))

    poligonos = sorted(HA_TODOS)
    vecinos = {h: [k for k in poligonos if frozenset((h, k)) in pares] for h in poligonos}
    no_vecinos = [(h, k) for i, h in enumerate(poligonos) for k in poligonos[i + 1:]
                  if frozenset((h, k)) not in pares]

    # huella de los CSV: si cambian, las anclas guardadas dejan de valer
    huella = hashlib.md5()
    for nombre in ('adyacencia.csv', 'supervivencia.csv', 'captura.csv'):
        with open(_ruta(nombre), 'rb') as f:
            huella.update(f.read())

    return {'adyacentes': pares, 'vecinos': vecinos, 'no_vecinos': no_vecinos,
            'pi': pi, 'cap': cap, 'poligonos': poligonos, 'huella': huella.hexdigest()}


def diagnostico(datos):
    """Revisa que el escenario pueda ser factible antes de resolver."""
    problemas = []
    dias_siembra = [t for t in cfg.DIAS if t >= cfg.PRIMER_DIA_PLANTABLE]
    ha_total = sum(HA_TODOS.values())
    ha_max_dia = cfg.CAP_TRAB * cfg.TRABAJADORES / ((1 - cfg.BANDA) * cfg.DENS_TOTAL)
    if cfg.HA_MIN_DIA * len(dias_siembra) > ha_total:
        problemas.append(f'Piden {cfg.HA_MIN_DIA*len(dias_siembra):.0f} ha en {len(dias_siembra)} días y solo hay {ha_total:.1f} ha')
    if cfg.HA_MIN_DIA > ha_max_dia:
        problemas.append(f'La cuadrilla solo cubre {ha_max_dia:.2f} ha/día y se exigen {cfg.HA_MIN_DIA}')
    sin_vecino = [h for h in datos['poligonos'] if not datos['vecinos'][h]]
    if sin_vecino:
        problemas.append(f'Polígonos sin ningún vecino en la adyacencia: {sin_vecino}')
    print(f'Modo: {cfg.MODO} | resultados en {cfg.CARPETA_RESULTADOS}')
    print(f'Capacidad: hasta {ha_max_dia:.2f} ha/día; mínimo exigido {cfg.HA_MIN_DIA} ha/día')
    print(f'Se necesitan entre {cfg.HA_MIN_DIA*len(dias_siembra):.0f} y {ha_max_dia*len(dias_siembra):.0f} ha '
          f'(hay {ha_total:.1f} ha) | compras con llegada entre el día 1 y el {cfg.ULTIMO_DIA_LLEGADA}')
    print('Chequeo previo:', 'sin problemas evidentes.' if not problemas else '')
    for p in problemas:
        print('  -', p)
    return problemas



# =====================================================================================
# Construcción del modelo
# =====================================================================================
GRUPO_INVENTARIO = ['restr_viajes', 'restr_balance_entrada', 'restr_balance_intermedio', 'restr_balance_limite',
                    'restr_bodega', 'restr_oferta', 'restr_mano_obra', 'restr_reparto',
                    'restr_poldia_min', 'restr_poldia_max', 'restr_especie_min', 'restr_especie_max']


def construir_modelo(datos):
    EDAD_MAX = cfg.EDAD_MAX
    DENS_TOTAL, BANDA, CAP_TRAB = cfg.DENS_TOTAL, cfg.BANDA, cfg.CAP_TRAB
    vecinos = datos['vecinos']

    m = pyo.ConcreteModel(name='Modelo1_SeleccionPoligonos')

    # ---------- Conjuntos ----------
    m.j = pyo.Set(initialize=VIVEROS)
    m.s = pyo.Set(initialize=ESPECIES)
    m.h = pyo.Set(initialize=datos['poligonos'])                 # polígonos candidatos
    m.t = pyo.Set(initialize=cfg.DIAS, ordered=True)
    m.a = pyo.Set(initialize=range(0, EDAD_MAX + 1))             # edades en bodega 0..7
    m.A_P = pyo.Set(initialize=list(cfg.EDADES_PLANTABLES))      # edades plantables 3..7
    m.SJ = pyo.Set(initialize=list(COSTOS_ESPECIES.keys()), dimen=2)   # pares (especie, vivero) válidos
    m.NV = pyo.Set(initialize=datos['no_vecinos'], dimen=2)      # pares h<k que NO pueden coincidir el mismo día
    m.TT = pyo.Set(initialize=[t for t in cfg.DIAS if t > cfg.PRIMER_DIA_PLANTABLE])

    # ---------- Parámetros ----------
    m.c_esp = pyo.Param(m.SJ, initialize=COSTOS_ESPECIES)
    m.o_esp = pyo.Param(m.SJ, initialize=OFERTA_ESPECIES)
    m.dens = pyo.Param(m.s, initialize=DENS)
    m.Ha = pyo.Param(m.h, initialize=HA_TODOS)
    m.W = pyo.Param(m.t, initialize={t: cfg.TRABAJADORES for t in cfg.DIAS})
    m.pi = pyo.Param(m.s, m.A_P, initialize=datos['pi'])          # supervivencia por especie y edad al plantar
    m.cap = pyo.Param(m.s, initialize=datos['cap'])               # carbono por planta sobreviviente

    # ---------- Variables ----------
    m.x = pyo.Var(m.SJ, m.t, domain=pyo.NonNegativeIntegers)        # compra de s a j que llega el día t
    m.n_t = pyo.Var(m.t, domain=pyo.NonNegativeIntegers)            # viajes de camión el día t
    m.I = pyo.Var(m.s, m.a, m.t, domain=pyo.NonNegativeIntegers)    # inventario de s con edad a al cierre del día t
    m.P = pyo.Var(m.s, m.A_P, m.t, domain=pyo.NonNegativeIntegers)  # plantas de s plantadas con edad a el día t
    m.M = pyo.Var(m.s, m.t, domain=pyo.NonNegativeIntegers)         # merma (cumple 7 días sin plantarse)
    m.Q = pyo.Var(m.s, m.h, m.t, domain=pyo.NonNegativeIntegers)    # plantas de s en el polígono h el día t
    m.ha = pyo.Var(m.h, m.t, domain=pyo.NonNegativeReals)           # ha del polígono h sembradas el día t
    m.y = pyo.Var(m.h, domain=pyo.Binary)                           # 1 si el polígono h se planta
    m.z = pyo.Var(m.h, m.t, domain=pyo.Binary)                      # 1 si el polígono h se trabaja el día t

    # ---------- Compras, bodega y envejecimiento ----------
    m.restr_viajes = pyo.Constraint(m.t, rule=lambda m, t:
        sum(m.x[s, j, t] for (s, j) in m.SJ) <= cfg.K_CAM * m.n_t[t])

    m.restr_balance_entrada = pyo.Constraint(m.s, m.t, rule=lambda m, s, t:
        m.I[s, 0, t] == sum(m.x[s2, j, t] for (s2, j) in m.SJ if s2 == s))

    # inventario de edad a hoy = el de edad a-1 ayer - lo plantado hoy (inventario inicial = 0)
    def balance_intermedio(m, s, a, t):
        previo = 0 if t == 1 else m.I[s, a - 1, t - 1]
        sale = m.P[s, a, t] if a in m.A_P else 0
        return m.I[s, a, t] == previo - sale
    m.restr_balance_intermedio = pyo.Constraint(m.s, pyo.RangeSet(1, EDAD_MAX), m.t, rule=balance_intermedio)

    # lo que queda con edad 7 al cierre no se planta: es merma
    m.restr_balance_limite = pyo.Constraint(m.s, m.t, rule=lambda m, s, t: m.M[s, t] == m.I[s, EDAD_MAX, t])

    m.restr_bodega = pyo.Constraint(m.t, rule=lambda m, t:
        sum(cfg.AREA_S * m.I[s, a, t] for s in m.s for a in range(0, EDAD_MAX)) <= cfg.K_ALM)

    m.restr_oferta = pyo.Constraint(m.SJ, m.t, rule=lambda m, s, j, t: m.x[s, j, t] <= m.o_esp[s, j])

    # ---------- Mano de obra ----------
    m.restr_mano_obra = pyo.Constraint(m.t, rule=lambda m, t:
        sum(m.P[s, a, t] for s in m.s for a in m.A_P) <= CAP_TRAB * m.W[t])

    # tope de ha/día que implica la cuadrilla (plantando el 90% de la densidad); refuerza la relajación
    m.restr_ha_max_dia = pyo.Constraint(m.t, rule=lambda m, t:
        sum(m.ha[h, t] for h in m.h) <= CAP_TRAB * m.W[t] / ((1 - BANDA) * DENS_TOTAL))

    # ---------- Selección de polígonos ----------
    m.restr_reparto = pyo.Constraint(m.s, m.t, rule=lambda m, s, t:
        sum(m.P[s, a, t] for a in m.A_P) == sum(m.Q[s, h, t] for h in m.h))
    m.restr_z_y = pyo.Constraint(m.h, m.t, rule=lambda m, h, t: m.z[h, t] <= m.y[h])
    m.restr_ha_z = pyo.Constraint(m.h, m.t, rule=lambda m, h, t: m.ha[h, t] <= m.Ha[h] * m.z[h, t])
    m.restr_lote_min = pyo.Constraint(m.h, m.t, rule=lambda m, h, t: m.ha[h, t] >= cfg.HA_LOTE_MIN * m.z[h, t])
    m.restr_cierre = pyo.Constraint(m.h, rule=lambda m, h: sum(m.ha[h, t] for t in m.t) == m.Ha[h] * m.y[h])
    m.restr_ha_min_dia = pyo.Constraint([t for t in cfg.DIAS if t >= cfg.PRIMER_DIA_PLANTABLE], rule=lambda m, t:
        sum(m.ha[h, t] for h in m.h) >= cfg.HA_MIN_DIA)

    # plantas del polígono h el día t: 524 por ha sembrada ese día (+-10%)
    m.restr_poldia_min = pyo.Constraint(m.h, m.t, rule=lambda m, h, t:
        sum(m.Q[s, h, t] for s in m.s) >= (1 - BANDA) * DENS_TOTAL * m.ha[h, t])
    m.restr_poldia_max = pyo.Constraint(m.h, m.t, rule=lambda m, h, t:
        sum(m.Q[s, h, t] for s in m.s) <= (1 + BANDA) * DENS_TOTAL * m.ha[h, t])

    # balance por especie al cerrar cada polígono (+-10%)
    m.restr_especie_min = pyo.Constraint(m.s, m.h, rule=lambda m, s, h:
        sum(m.Q[s, h, t] for t in m.t) >= (1 - BANDA) * m.dens[s] * m.Ha[h] * m.y[h])
    m.restr_especie_max = pyo.Constraint(m.s, m.h, rule=lambda m, s, h:
        sum(m.Q[s, h, t] for t in m.t) <= (1 + BANDA) * m.dens[s] * m.Ha[h] * m.y[h])

    # ---------- Adyacencia (continuidad de siembra) ----------
    # Regla 1: el mismo día, todos los polígonos que se trabajan son vecinos entre sí
    m.restr_ady_mismo_dia = pyo.Constraint(m.NV, m.t, rule=lambda m, h, k, t: m.z[h, t] + m.z[k, t] <= 1)
    # Regla 2: h solo se trabaja el día t si él o algún vecino se trabajó el día t-1
    m.restr_ady_avance = pyo.Constraint(m.h, m.TT, rule=lambda m, h, t:
        m.z[h, t] <= m.z[h, t - 1] + sum(m.z[k, t - 1] for k in vecinos[h]))

    # ---------- Fijaciones ----------
    for h in m.h:                                   # antes del primer día plantable no se siembra (aclimatación)
        for t in cfg.DIAS:
            if t < cfg.PRIMER_DIA_PLANTABLE:
                m.z[h, t].fix(0)
    for (s, j) in m.SJ:                             # lo que llega después de N_DIAS - 3 no alcanza edad plantable
        for t in cfg.DIAS:
            if t > cfg.ULTIMO_DIA_LLEGADA:
                m.x[s, j, t].fix(0)

    # ---------- Objetivos ----------
    m.costo_compra = pyo.Expression(expr=sum(m.c_esp[s, j] * m.x[s, j, t] for (s, j) in m.SJ for t in m.t))
    m.costo_camion = pyo.Expression(expr=sum(cfg.C_CAM * m.n_t[t] for t in m.t))
    m.costo_siembra = pyo.Expression(expr=sum(cfg.C_PLANTAR * m.P[s, a, t] for s in m.s for a in m.A_P for t in m.t))
    m.costo_merma = pyo.Expression(expr=sum(cfg.C_MERMA * m.M[s, t] for s in m.s for t in m.t))
    m.costo = pyo.Expression(expr=m.costo_compra + m.costo_camion + m.costo_siembra + m.costo_merma)


    # carbono = captura de la especie x supervivencia según días en bodega x plantas sembradas
    m.carbono = pyo.Expression(expr=
        sum(m.cap[s] * m.pi[s, a] * m.P[s, a, t] for s in m.s for a in m.A_P for t in m.t))

    # ---------- Componentes de AUGMECON ----------
    # Barrido maxCarbono:  max carbono + rho * r_carb * s_costo / r_costo   s.a.  costo   + s_costo == eps_costo
    # Barrido minCosto:    min costo   - rho * r_costo * s_carb / r_carb    s.a.  carbono - s_carb  == eps_carb
    # (r_* = rangos de la tabla de pagos; el término aumentado evita puntos débilmente eficientes)
    m.s_costo = pyo.Var(domain=pyo.NonNegativeReals)
    m.s_carb = pyo.Var(domain=pyo.NonNegativeReals)
    m.eps_costo = pyo.Param(initialize=1e12, mutable=True)
    m.eps_carb = pyo.Param(initialize=0.0, mutable=True)
    m.rango_costo = pyo.Param(initialize=1.0, mutable=True)
    m.rango_carb = pyo.Param(initialize=1.0, mutable=True)
    m.rho = pyo.Param(initialize=cfg.RHO, mutable=True)

    m.lim_costo = pyo.Constraint(expr=m.costo + m.s_costo == m.eps_costo)
    m.lim_carb = pyo.Constraint(expr=m.carbono - m.s_carb == m.eps_carb)
    m.lim_costo.deactivate()
    m.lim_carb.deactivate()

    m.obj_ha = pyo.Objective(expr=sum(m.ha[h, t] for h in m.h for t in m.t), sense=pyo.minimize)   # solo etapa 0
    m.obj_costo = pyo.Objective(expr=m.costo, sense=pyo.minimize)
    m.obj_carbono = pyo.Objective(expr=m.carbono, sense=pyo.maximize)
    m.obj_aug_maxCarbono = pyo.Objective(
        expr=m.carbono + m.rho * m.rango_carb * m.s_costo / m.rango_costo, sense=pyo.maximize)
    m.obj_aug_minCosto = pyo.Objective(
        expr=m.costo - m.rho * m.rango_costo * m.s_carb / m.rango_carb, sense=pyo.minimize)
    for o in m.component_objects(pyo.Objective):
        o.deactivate()

    m._datos = datos
    return m


# =====================================================================================
# Utilidades de resolución
# =====================================================================================
def activar_inventario(m, flag):
    for nombre in GRUPO_INVENTARIO:
        c = getattr(m, nombre)
        c.activate() if flag else c.deactivate()


def seleccion_actual(m):
    y = {h: round(pyo.value(m.y[h])) for h in m.h}
    z = {(h, t): round(pyo.value(m.z[h, t])) for h in m.h for t in m.t if t >= cfg.PRIMER_DIA_PLANTABLE}
    return y, z


def fijar_seleccion(m, y, z):
    for h in m.h:
        m.y[h].fix(y[h])
        for t in m.t:
            if t >= cfg.PRIMER_DIA_PLANTABLE:
                m.z[h, t].fix(z[h, t])


def liberar_seleccion(m):
    for h in m.h:
        m.y[h].unfix()
        for t in m.t:
            if t >= cfg.PRIMER_DIA_PLANTABLE:
                m.z[h, t].unfix()


def _gap(res):
    try:
        lb, ub = float(res.problem.lower_bound), float(res.problem.upper_bound)
        if math.isinf(lb) or math.isinf(ub) or math.isnan(lb) or math.isnan(ub):
            return None
        return abs(ub - lb) / max(abs(ub), abs(lb), 1e-9)
    except Exception:
        return None


def resolver(m, objetivo, tiempo=None, warm=False, primera=False, etiqueta=''):
    """objetivo: nombre del Objective a activar. warm=True arranca desde los valores actuales de las variables.
    Si el solver no termina en 'optimal', reintenta con más tiempo (cfg.REINTENTOS veces, multiplicando el límite
    por cfg.FACTOR_TIEMPO) retomando desde la solución actual. Devuelve dict con ok, estado, gap y segundos."""
    for o in m.component_objects(pyo.Objective):
        o.deactivate()
    getattr(m, objetivo).activate()
    solver = pyo.SolverFactory(cfg.SOLVER)
    solver.options['mip_rel_gap'] = cfg.GAP
    if primera:
        solver.options['mip_max_improving_sols'] = 1   # se detiene en la primera solución factible
    limite = tiempo or cfg.TIEMPO_PUNTO
    n_reint = 0 if primera else cfg.REINTENTOS         # la etapa 0 se detiene a propósito: no se reintenta
    t0 = time.time()
    for intento in range(n_reint + 1):
        solver.options['time_limit'] = limite * cfg.FACTOR_TIEMPO ** intento
        try:
            res = solver.solve(m, tee=False, warmstart=(warm or intento > 0))
        except Exception as e:
            seg = time.time() - t0
            print(f'[{etiqueta or objetivo}] sin solución ({seg:.0f} s):', str(e)[:150])
            return {'ok': False, 'estado': 'sin_solucion', 'gap': None, 'segundos': round(seg, 1)}
        if str(res.solver.termination_condition) == 'optimal':
            break
        if intento < n_reint:
            print(f'[{etiqueta or objetivo}] {res.solver.termination_condition}: reintento {intento + 1} '
                  f'con {limite * cfg.FACTOR_TIEMPO ** (intento + 1):.0f} s')
    seg = time.time() - t0
    estado = str(res.solver.termination_condition)
    gap = _gap(res)
    gap_txt = f'{gap:.2%}' if gap is not None else 'n/d'
    # en la etapa 0 las compras están fuera del problema y no tienen valor: no se evalúa el costo
    costo = pyo.value(m.costo, exception=False)
    carbono = pyo.value(m.carbono, exception=False)
    fmt = lambda v: f'{v:,.1f}' if v is not None else 'n/d'
    print(f'[{etiqueta or objetivo}] {estado} | costo = {fmt(costo)} | '
          f'carbono = {fmt(carbono)} | gap = {gap_txt} | {seg:.0f} s')
    return {'ok': True, 'estado': estado, 'gap': gap, 'segundos': round(seg, 1)}


def resolver_dos_fases(m, objetivo, etiqueta=''):
    """Fase 1: fija la selección de polígonos que traen las variables (punto de arranque) y resuelve; así el
    solver sí mejora en poco tiempo. Fase 2 (solo si LIBERAR_SELECCION): libera la selección y sigue
    desde la fase 1. Si la fase 2 no termina con solución, se conserva la fase 1.
    Requiere que las variables tengan una solución factible cargada (warm start)."""
    y, z = seleccion_actual(m)
    fijar_seleccion(m, y, z)
    info1 = resolver(m, objetivo, tiempo=cfg.TIEMPO_FIJA, warm=True, etiqueta=f'{etiqueta} [fase 1: selección fija]')
    if not cfg.LIBERAR_SELECCION:
        return info1                                   # en selFija la selección ya estaba fija; se queda así
    liberar_seleccion(m)
    obj = getattr(m, objetivo)
    signo = 1 if obj.sense == pyo.minimize else -1             # para comparar "mejor" en ambos sentidos
    foto1 = capturar(m) if info1['ok'] else None
    val1 = signo * pyo.value(obj, exception=False) if info1['ok'] else None
    info2 = resolver(m, objetivo, tiempo=cfg.TIEMPO_PUNTO, warm=True, etiqueta=f'{etiqueta} [fase 2: selección libre]')
    val2 = signo * pyo.value(obj, exception=False) if info2['ok'] else None
    if info2['ok'] and (val1 is None or val2 <= val1 + 1e-6 * max(1.0, abs(val1))):
        info2['segundos'] = round(info2['segundos'] + info1['segundos'], 1)
        return info2
    if foto1 is not None:                                   # la fase 2 no mejoró: se queda la fase 1
        restaurar(m, foto1)
    return info1


def capturar(m):
    """Foto de todas las variables (para warm start y para reconstruir planes)."""
    return {v.name: {i: v[i].value for i in v} for v in m.component_objects(pyo.Var)}


def restaurar(m, foto):
    for v in m.component_objects(pyo.Var):
        valores = foto.get(v.name, {})
        for i, val in valores.items():
            if i in v:
                v[i].set_value(val, skip_validation=True)


def _solucion(m, info, nombre):
    return {'nombre': nombre, 'costo': pyo.value(m.costo), 'carbono': pyo.value(m.carbono),
            'foto': capturar(m), **{k: info[k] for k in ('estado', 'gap', 'segundos')}}


# =====================================================================================
# Tabla de pagos lexicográfica (anclas)
# =====================================================================================
def _clave_anclas(datos):
    conf = {k: v for k, v in asdict(cfg).items() if k not in _NO_AFECTAN_ANCLAS}
    return hashlib.md5((repr(sorted(conf.items())) + datos['huella'] + f'v{VERSION_ANCLAS}').encode()).hexdigest()


def obtener_anclas(m, datos, recalcular=False):
    """Calcula (o carga) las dos anclas lexicográficas:
        A (barato): min costo -> C_min;  luego max carbono s.a. costo <= C_min
        B (verde):  max carbono -> G_max; luego min costo s.a. carbono >= G_max
    Se guardan en resultados_<modo>/anclas/anclas.pkl y las reutilizan ambos barridos."""
    carpeta = os.path.join(cfg.CARPETA_RESULTADOS, 'anclas')
    os.makedirs(carpeta, exist_ok=True)
    ruta_pkl = os.path.join(carpeta, 'anclas.pkl')
    clave = _clave_anclas(datos)

    if os.path.exists(ruta_pkl) and not recalcular:
        with open(ruta_pkl, 'rb') as f:
            anclas = pickle.load(f)
        if anclas.get('clave') == clave:
            print(f'Anclas cargadas de {ruta_pkl} (mismos datos y parámetros).')
            _aplicar_modo(m, anclas)
            _mostrar_tabla_pagos(anclas)
            return anclas
        print('Los datos o parámetros cambiaron desde que se guardaron las anclas: se recalculan.')

    registro = []
    def reg(info, paso):
        registro.append({'paso': paso, 'costo': pyo.value(m.costo, exception=False),
                         'carbono': pyo.value(m.carbono, exception=False), **info})

    m.lim_costo.deactivate(); m.lim_carb.deactivate()
    liberar_seleccion(m)

    # Etapa 0: calendario factible (sin compras ni bodega), se detiene en la primera solución
    activar_inventario(m, False)
    info = resolver(m, 'obj_ha', tiempo=cfg.TIEMPO_ETAPA0, primera=True, etiqueta='etapa0 calendario')
    activar_inventario(m, True)
    if not info['ok']:
        raise RuntimeError('No se encontró un calendario factible en la etapa 0.')
    y0, z0 = seleccion_actual(m)

    # Costo mínimo con el calendario de la etapa 0 fijo -> primer plan completo
    fijar_seleccion(m, y0, z0)
    info = resolver(m, 'obj_costo', etiqueta='costo min (calendario etapa0 fijo)')
    liberar_seleccion(m)
    if not info['ok']:
        raise RuntimeError('No se encontró un plan completo con el calendario de la etapa 0.')
    reg(info, 'costo_min_calendario_etapa0')

    # Costo mínimo con selección libre, arrancando desde el plan anterior -> C_min
    info = resolver(m, 'obj_costo', warm=True, etiqueta='ancla A paso 1: min costo')
    reg(info, 'A1_min_costo')
    C_min = pyo.value(m.costo)

    # Modo selección fija: a partir de aquí se fija el calendario del plan de costo mínimo
    seleccion = seleccion_actual(m)
    if not cfg.LIBERAR_SELECCION:
        fijar_seleccion(m, *seleccion)

    # Ancla A paso 2: max carbono s.a. costo <= C_min
    m.eps_costo = C_min * (1 + 1e-9) + 1e-6
    m.s_costo.set_value(max(0.0, pyo.value(m.eps_costo) - C_min))
    m.lim_costo.activate()
    info = resolver_dos_fases(m, 'obj_carbono', etiqueta='ancla A paso 2: max carbono | costo <= C_min')
    m.lim_costo.deactivate()
    reg(info, 'A2_max_carbono_con_costo_min')
    A = _solucion(m, info, 'A')

    # Ancla B paso 1: max carbono sin tope de costo (arranca desde A)
    info = resolver_dos_fases(m, 'obj_carbono', etiqueta='ancla B paso 1: max carbono')
    reg(info, 'B1_max_carbono')
    G_max = pyo.value(m.carbono)

    # Ancla B paso 2: min costo s.a. carbono >= G_max
    m.eps_carb = G_max - 1e-7 * max(1.0, abs(G_max))
    m.s_carb.set_value(max(0.0, G_max - pyo.value(m.eps_carb)))
    m.lim_carb.activate()
    info = resolver_dos_fases(m, 'obj_costo', etiqueta='ancla B paso 2: min costo | carbono >= G_max')
    m.lim_carb.deactivate()
    reg(info, 'B2_min_costo_con_carbono_max')
    B = _solucion(m, info, 'B')
    if B['carbono'] - A['carbono'] <= 1e-6 * max(1.0, abs(A['carbono'])):
        print('AVISO: el ancla B no mejoró el carbono del ancla A. Sube TIEMPO_FIJA / TIEMPO_PUNTO o revisa los '
              'datos de supervivencia y captura; con anclas iguales no hay frontera.')

    anclas = {'clave': clave, 'modo': cfg.MODO, 'A': A, 'B': B, 'seleccion': seleccion}
    with open(ruta_pkl, 'wb') as f:
        pickle.dump(anclas, f)

    pd.DataFrame(registro).to_csv(os.path.join(carpeta, 'registro_anclas.csv'), index=False, encoding='utf-8-sig')
    _mostrar_tabla_pagos(anclas, guardar=True)
    for nombre in ('A', 'B'):                       # plan completo de cada ancla
        restaurar(m, anclas[nombre]['foto'])
        exportar_plan(m, carpeta, f'ancla{nombre}')
    return anclas


def _aplicar_modo(m, anclas):
    if cfg.LIBERAR_SELECCION:
        liberar_seleccion(m)
    else:
        fijar_seleccion(m, *anclas['seleccion'])


def _mostrar_tabla_pagos(anclas, guardar=False):
    A, B = anclas['A'], anclas['B']
    tabla = pd.DataFrame([
        {'ancla': 'A (min costo, luego max carbono)', 'costo': A['costo'], 'carbono': A['carbono'],
         'estado': A['estado'], 'gap': A['gap']},
        {'ancla': 'B (max carbono, luego min costo)', 'costo': B['costo'], 'carbono': B['carbono'],
         'estado': B['estado'], 'gap': B['gap']},
    ])
    print(f"Tabla de pagos ({anclas['modo']}): rango costo = {B['costo'] - A['costo']:,.1f} | "
          f"rango carbono = {B['carbono'] - A['carbono']:,.1f}")
    try:
        from IPython.display import display
        display(tabla)
    except ImportError:
        print(tabla)
    if guardar:
        tabla.to_csv(os.path.join(cfg.CARPETA_RESULTADOS, 'tabla_pagos.csv'), index=False, encoding='utf-8-sig')
    return tabla


# =====================================================================================
# Barrido ε-constraint (AUGMECON2)
# =====================================================================================
DIRECCIONES = {
    # nombre: (restricción, objetivo aumentado, holgura, parámetro ε, objetivo que lleva ε)
    'maxCarbono': ('lim_costo', 'obj_aug_maxCarbono', 's_costo', 'eps_costo', 'costo'),
    'minCosto':   ('lim_carb',  'obj_aug_minCosto',   's_carb',  'eps_carb',  'carbono'),
}


def barrido(m, anclas, direccion):
    """direccion = 'maxCarbono': max carbono s.a. costo <= ε,   ε en [costo_A, costo_B]
       direccion = 'minCosto':   min costo   s.a. carbono >= ε, ε en [carbono_A, carbono_B]
    La malla de ε sale de la tabla de pagos (N_PUNTOS intervalos); k=0 es el ancla A y k=N el ancla B.
    Guarda frontera_<direccion>.csv y el plan completo de cada punto en barrido_<direccion>/."""
    assert direccion in DIRECCIONES, f'direccion debe ser una de {list(DIRECCIONES)}'
    if anclas.get('modo') != cfg.MODO:
        raise RuntimeError(
            f"Las anclas son del modo '{anclas.get('modo')}' pero cfg está en '{cfg.MODO}'. "
            "Probablemente se reinició la configuración (importlib.reload o una celda con otro LIBERAR_SELECCION). "
            "Vuelve a correr la celda de configuración y la de anclas antes del barrido.")
    print(f'Barrido {direccion} | modo {cfg.MODO} | resultados en {cfg.CARPETA_RESULTADOS}')
    restr, objetivo, holgura, eps_nombre, f_eps = DIRECCIONES[direccion]
    A, B = anclas['A'], anclas['B']
    r_costo, r_carb = B['costo'] - A['costo'], B['carbono'] - A['carbono']
    N = cfg.N_PUNTOS
    carpeta = os.path.join(cfg.CARPETA_RESULTADOS, f'barrido_{direccion}')
    os.makedirs(carpeta, exist_ok=True)
    _aplicar_modo(m, anclas)

    if r_costo <= 1e-6 * max(1, abs(A['costo'])) or r_carb <= 1e-6 * max(1, abs(A['carbono'])):
        print(f'AVISO: rango de costo = {r_costo:,.2f}, rango de carbono = {r_carb:,.2f}. '
              'Los objetivos no están en conflicto (o el ancla B no mejoró al ancla A en el tiempo límite): '
              'la frontera es un solo punto, el ancla A.')
        restaurar(m, A['foto'])
        res = exportar_plan(m, carpeta, f'{direccion}_p00')
        frontera = pd.DataFrame([{'barrido': direccion, 'punto': 'p00', 'k': 0, 'epsilon': None,
                                  'costo': A['costo'], 'carbono': A['carbono'], 'estado': A['estado'],
                                  'gap': A['gap'], 'segundos': A['segundos'], 'nota': 'ancla A (sin conflicto)',
                                  'dominado': False, 'duplicado': False, **res}])
        ruta = os.path.join(cfg.CARPETA_RESULTADOS, f'frontera_{direccion}.csv')
        frontera.to_csv(ruta, index=False, encoding='utf-8-sig')
        print(f'Guardado: {ruta}')
        return frontera

    m.rango_costo = r_costo
    m.rango_carb = r_carb
    m.rho = cfg.RHO
    lo, hi = (A['costo'], B['costo']) if direccion == 'maxCarbono' else (A['carbono'], B['carbono'])
    paso = (hi - lo) / N
    malla = {k: lo + k * paso for k in range(N + 1)}
    # orden de AUGMECON2: de la restricción más laxa a la más estricta, para que el bypass sea válido
    orden = list(range(N - 1, 0, -1)) if direccion == 'maxCarbono' else list(range(1, N))

    def factible_para(sol, e):     # ¿esta solución cumple la restricción con este ε? (sirve de arranque)
        return sol['costo'] <= e + 1e-6 if direccion == 'maxCarbono' else sol['carbono'] >= e - 1e-6

    def mejor(cands):              # la mejor según el objetivo del barrido
        return max(cands, key=lambda s: s['carbono']) if direccion == 'maxCarbono' else min(cands, key=lambda s: s['costo'])

    filas, soluciones = [], [A, B]

    def fila(k, sol, estado, gap, seg, holg, warm, nota=''):
        prefijo = f'{direccion}_p{k:02d}'
        return {'barrido': direccion, 'punto': f'p{k:02d}', 'k': k, 'epsilon': malla[k],
                'costo': sol['costo'], 'carbono': sol['carbono'], 'holgura': holg,
                'estado': estado, 'gap': gap, 'segundos': seg, 'arranque': warm,
                'archivos': prefijo if not nota else '', 'nota': nota}

    # extremos = anclas (no se vuelven a resolver)
    for k, sol in ((0, A), (N, B)):
        restaurar(m, sol['foto'])
        res = exportar_plan(m, carpeta, f'{direccion}_p{k:02d}')
        filas.append({**fila(k, sol, sol['estado'], sol['gap'], sol['segundos'], None, '-',
                             ), 'nota': f"ancla {sol['nombre']}", **res})

    def guardar_parcial():         # se reescribe después de cada punto: si se interrumpe, no se pierde lo hecho
        pd.DataFrame(filas).sort_values('k').to_csv(
            os.path.join(cfg.CARPETA_RESULTADOS, f'frontera_{direccion}_parcial.csv'), index=False, encoding='utf-8-sig')

    def registrar_bypass(k, sol, s_val):
        if cfg.USAR_BYPASS and s_val is not None:   # los siguientes b puntos dan la misma solución
            b = int(math.floor(s_val / paso + 1e-9))
            siguientes = range(k - 1, k - 1 - b, -1) if direccion == 'maxCarbono' else range(k + 1, k + 1 + b)
            for kk in siguientes:
                if kk in orden:
                    saltar[kk] = {'sol': sol, 'punto': f'p{k:02d}'}

    clave_malla = (round(lo, 6), round(hi, 6), N, cfg.RHO, cfg.MODO)   # un punto guardado solo vale con la misma malla
    saltar = {}
    getattr(m, restr).activate()
    for k in orden:
        e = malla[k]
        if k in saltar:                              # bypass AUGMECON2: mismo óptimo que un punto ya resuelto
            ref = saltar[k]
            filas.append(fila(k, ref['sol'], 'bypass', ref['sol']['gap'], 0, None, '-',
                              nota=f"igual a {ref['punto']} (bypass)"))
            print(f'[{direccion} p{k:02d}] ε = {e:,.1f} -> bypass, igual a {ref["punto"]}')
            guardar_parcial()
            continue

        ruta_pkl = os.path.join(carpeta, f'{direccion}_p{k:02d}.pkl')
        if cfg.REANUDAR and os.path.exists(ruta_pkl):  # punto ya resuelto en una corrida anterior
            with open(ruta_pkl, 'rb') as f:
                guardado = pickle.load(f)
            if guardado.get('clave_malla') == clave_malla:
                soluciones.append(guardado['sol'])
                filas.append(guardado['fila'])
                registrar_bypass(k, guardado['sol'], guardado['s_val'])
                print(f'[{direccion} p{k:02d}] ε = {e:,.1f} -> cargado de una corrida anterior')
                guardar_parcial()
                continue

        print(f'{time.strftime("%H:%M:%S")} inicia {direccion} p{k:02d}')
        arranque = mejor([s for s in soluciones if factible_para(s, e)])
        restaurar(m, arranque['foto'])
        setattr(m, eps_nombre, e)
        getattr(m, holgura).set_value(max(0.0, e - arranque['costo']) if direccion == 'maxCarbono'
                                      else max(0.0, arranque['carbono'] - e))
        info = resolver_dos_fases(m, objetivo, etiqueta=f'{direccion} p{k:02d} ε={e:,.1f}')
        if not info['ok']:
            filas.append({'barrido': direccion, 'punto': f'p{k:02d}', 'k': k, 'epsilon': e,
                          'estado': info['estado'], 'segundos': info['segundos'], 'arranque': arranque['nombre'],
                          'nota': 'sin solución en el tiempo límite'})
            guardar_parcial()
            continue

        sol = _solucion(m, info, f'p{k:02d}')
        soluciones.append(sol)
        s_val = max(0.0, pyo.value(getattr(m, holgura)))
        res = exportar_plan(m, carpeta, f'{direccion}_p{k:02d}')
        fila_k = {**fila(k, sol, info['estado'], info['gap'], info['segundos'], s_val, arranque['nombre']), **res}
        filas.append(fila_k)
        with open(ruta_pkl, 'wb') as f:
            pickle.dump({'clave_malla': clave_malla, 'sol': sol, 'fila': fila_k, 's_val': s_val}, f)
        guardar_parcial()
        registrar_bypass(k, sol, s_val)
    getattr(m, restr).deactivate()

    frontera = pd.DataFrame(filas).sort_values('k').reset_index(drop=True)
    frontera = marcar_dominados(frontera)
    ruta = os.path.join(cfg.CARPETA_RESULTADOS, f'frontera_{direccion}.csv')
    frontera.to_csv(ruta, index=False, encoding='utf-8-sig')
    print(f'Guardado: {ruta}')
    return frontera


def marcar_dominados(df, tol=1e-6):
    """dominado = existe otro punto con costo <= y carbono >= (y uno de los dos estrictamente mejor).
    duplicado = mismo (costo, carbono) que un punto anterior (bypass o soluciones repetidas)."""
    df = df.copy()
    validos = df['costo'].notna() & df['carbono'].notna()
    dom, dup = [], []
    vistos = []
    for i, r in df.iterrows():
        if not validos[i]:
            dom.append(None); dup.append(None); continue
        es_dom = any((o.costo <= r.costo + tol and o.carbono >= r.carbono - tol) and
                     (o.costo < r.costo - tol or o.carbono > r.carbono + tol)
                     for _, o in df[validos].iterrows())
        es_dup = any(abs(c - r.costo) <= tol and abs(g - r.carbono) <= tol for c, g in vistos)
        vistos.append((r.costo, r.carbono))
        dom.append(es_dom); dup.append(es_dup)
    df['dominado'] = dom
    df['duplicado'] = dup
    return df


def combinar_fronteras():
    """Une frontera_maxCarbono.csv y frontera_minCosto.csv del modo actual, quita dominados y duplicados,
    y calcula el costo marginal por unidad de carbono y el punto rodilla."""
    print(f'Combinando fronteras del modo {cfg.MODO} en {cfg.CARPETA_RESULTADOS}')
    partes, faltan = [], []
    for d in DIRECCIONES:
        ruta = os.path.join(cfg.CARPETA_RESULTADOS, f'frontera_{d}.csv')
        if os.path.exists(ruta):
            partes.append(pd.read_csv(ruta, encoding='utf-8-sig'))
        else:
            faltan.append(ruta)
    if faltan:
        print('No se pueden combinar las fronteras; no existe:')
        for r in faltan:
            print('  -', r)
        print('Revisa que ese barrido haya terminado (celda del barrido) y que ambos notebooks usen '
              'la misma CARPETA y el mismo LIBERAR_SELECCION.')
        return None
    df = pd.concat(partes, ignore_index=True)
    df = df[df['costo'].notna() & (df['estado'] != 'bypass')].drop(columns=['dominado', 'duplicado'])
    df = marcar_dominados(df.sort_values(['costo', 'carbono'], ascending=[True, False]).reset_index(drop=True))
    df = df[~df['dominado'] & ~df['duplicado']].sort_values('costo').reset_index(drop=True)

    df['costo_marginal'] = df['costo'].diff() / df['carbono'].diff()   # $ por unidad adicional de carbono
    # rodilla: punto más alejado de la recta entre los extremos (en escala normalizada)
    c = (df['costo'] - df['costo'].min()) / max(df['costo'].max() - df['costo'].min(), 1e-9)
    g = (df['carbono'] - df['carbono'].min()) / max(df['carbono'].max() - df['carbono'].min(), 1e-9)
    dist = (g - c).abs() / math.sqrt(2)
    df['rodilla'] = False
    if len(df) > 2:
        df.loc[dist.idxmax(), 'rodilla'] = True

    ruta = os.path.join(cfg.CARPETA_RESULTADOS, 'frontera_combinada.csv')
    df.to_csv(ruta, index=False, encoding='utf-8-sig')
    print(f'Guardado: {ruta} ({len(df)} puntos no dominados)')
    return df


# =====================================================================================
# Extracción y exportación de planes
# =====================================================================================
def extraer_calendario(m, tol=1e-6):
    filas = [(t, h, round(pyo.value(m.ha[h, t]), 6)) for h in m.h for t in m.t if pyo.value(m.ha[h, t]) > tol]
    return pd.DataFrame(filas, columns=['dia', 'poligono', 'ha']).sort_values(['dia', 'poligono']).reset_index(drop=True)


def extraer_pedidos(m):
    filas = [{'Especie': s, 'Comprar a': j, 'Día de pedido': t - 1, 'Día de llegada': t,
              'Cantidad': round(pyo.value(m.x[s, j, t]))}
             for (s, j) in m.SJ for t in m.t if pyo.value(m.x[s, j, t]) > 0.5]
    df = pd.DataFrame(filas, columns=['Especie', 'Comprar a', 'Día de pedido', 'Día de llegada', 'Cantidad'])
    return df.sort_values(['Especie', 'Comprar a', 'Día de pedido']).reset_index(drop=True)


def verificar_adyacencia(m, cal):
    """(día, h, k): activos el mismo día sin ser vecinos; (día, h): activo sin él o un vecino activo el día anterior."""
    ady, vecinos = m._datos['adyacentes'], m._datos['vecinos']
    activos = {t: set(cal[cal.dia == t].poligono) for t in cfg.DIAS}
    malos = []
    for t in cfg.DIAS:
        for h in activos[t]:
            for k in activos[t]:
                if h < k and frozenset((h, k)) not in ady:
                    malos.append((t, h, k))
            if t > cfg.PRIMER_DIA_PLANTABLE and not (activos[t - 1] & ({h} | set(vecinos[h]))):
                malos.append((t, h))
    return malos


def plan_siembra(m, tol=0.5):
    """Tabla dia, poligono, especie, lote, proveedor, dia_llegada, edad, plantas, supervivencia, sobrevivientes, carbono.
    Reparte edades entre polígonos (esquina noroeste) y consume los lotes de cada proveedor en orden."""
    lotes = {}
    for (s, j) in m.SJ:
        for d in m.t:
            q = round(pyo.value(m.x[s, j, d]))
            if q > 0:
                lotes.setdefault((s, d), []).append([j, q])
    filas = []
    for s in m.s:
        for t in m.t:
            oferta = [[a, round(pyo.value(m.P[s, a, t]))] for a in m.A_P if pyo.value(m.P[s, a, t]) > tol]
            demanda = [[h, round(pyo.value(m.Q[s, h, t]))] for h in m.h if pyo.value(m.Q[s, h, t]) > tol]
            if sum(n for _, n in oferta) != sum(n for _, n in demanda):
                raise ValueError(f'P y Q no coinciden en {s}, día {t}')
            i = k = 0
            while i < len(oferta) and k < len(demanda):
                a, hay = oferta[i]; h, falta = demanda[k]
                n = min(hay, falta)
                d = t - a                                   # día de llegada del lote
                restante = n
                while restante > 0:
                    lote = next(l for l in lotes[(s, d)] if l[1] > 0)
                    toma = min(restante, lote[1]); lote[1] -= toma; restante -= toma
                    filas.append({'dia_plantacion': t, 'poligono': int(h), 'especie': s,
                                  'lote': f'{s}|{lote[0]}|d{d}', 'proveedor': lote[0], 'dia_llegada': d, 'edad': a,
                                  'plantas': toma, 'supervivencia': m.pi[s, a],
                                  'sobrevivientes': toma * m.pi[s, a], 'carbono': toma * m.pi[s, a] * m.cap[s]})
                oferta[i][1] -= n; demanda[k][1] -= n
                if oferta[i][1] == 0: i += 1
                if demanda[k][1] == 0: k += 1
    return pd.DataFrame(filas).sort_values(['dia_plantacion', 'poligono', 'especie', 'edad']).reset_index(drop=True)


def resumen_plan(m):
    """Indicadores del plan actual: sirven para ver cómo cambian las decisiones a lo largo de la frontera."""
    plantas = sum(pyo.value(m.P[s, a, t]) for s in m.s for a in m.A_P for t in m.t)
    ultimo = cfg.DIAS[-1]
    ha_total = sum(pyo.value(m.ha[h, t]) for h in m.h for t in m.t)
    costos = {f'costo_{c}': pyo.value(getattr(m, f'costo_{c}')) for c in COMPONENTES_COSTO}
    return {
        'n_poligonos': sum(round(pyo.value(m.y[h])) for h in m.h),
        'ha_total': ha_total,
        **costos,
        'costo_por_ha': pyo.value(m.costo) / ha_total if ha_total else None,
        'comprado': sum(pyo.value(m.x[s, j, t]) for (s, j) in m.SJ for t in m.t),
        'plantado': plantas,
        'merma': sum(pyo.value(m.M[s, t]) for s in m.s for t in m.t),
        'sin_plantar_al_cierre': sum(pyo.value(m.I[s, a, ultimo]) for s in m.s for a in range(cfg.EDAD_MAX)),
        'viajes': sum(pyo.value(m.n_t[t]) for t in m.t),
        'edad_promedio': (sum(a * pyo.value(m.P[s, a, t]) for s in m.s for a in m.A_P for t in m.t) / plantas
                          if plantas else None),
        'supervivencia_promedio': (sum(pyo.value(m.pi[s, a]) * pyo.value(m.P[s, a, t])
                                       for s in m.s for a in m.A_P for t in m.t) / plantas if plantas else None),
    }


def exportar_plan(m, carpeta, prefijo):
    """Escribe <prefijo>_calendario.csv, <prefijo>_compras.csv y <prefijo>_plan_siembra.csv con la solución actual.
    Devuelve el resumen del plan (se agrega a la tabla de la frontera)."""
    os.makedirs(carpeta, exist_ok=True)
    resumen = resumen_plan(m)
    cal = extraer_calendario(m)
    pedidos = extraer_pedidos(m)
    cal.to_csv(os.path.join(carpeta, f'{prefijo}_calendario.csv'), index=False, encoding='utf-8-sig')
    pedidos.to_csv(os.path.join(carpeta, f'{prefijo}_compras.csv'), index=False, encoding='utf-8-sig')
    malos = verificar_adyacencia(m, cal)
    resumen['violaciones_adyacencia'] = len(malos)
    try:
        plan = plan_siembra(m)
        carbono_modelo = pyo.value(m.carbono)
        if abs(plan.carbono.sum() - carbono_modelo) > 1e-3 * max(1, carbono_modelo):
            print(f'  aviso {prefijo}: el carbono del plan ({plan.carbono.sum():,.1f}) no coincide con el modelo '
                  f'({carbono_modelo:,.1f})')
        plan.to_csv(os.path.join(carpeta, f'{prefijo}_plan_siembra.csv'), index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f'  aviso {prefijo}: no se pudo armar el plan de siembra por lote: {e}')
    if malos:
        print(f'  aviso {prefijo}: {len(malos)} violaciones de adyacencia: {malos[:5]}')
    return resumen


# =====================================================================================
# Decisión a posteriori sobre la frontera (suma ponderada, sensibilidad, TOPSIS, precio del carbono)
# =====================================================================================
def _soportados(costo, carbono):
    """Índices de los puntos en la envolvente cóncava superior (costo ascendente, carbono a maximizar).
    Solo estos puntos pueden ser elegidos por una suma ponderada, para algún peso."""
    orden = sorted(range(len(costo)), key=lambda i: (costo[i], -carbono[i]))
    H = []
    for i in orden:
        while len(H) >= 2:
            o, a = H[-2], H[-1]
            cruz = (costo[a] - costo[o]) * (carbono[i] - carbono[o]) - (carbono[a] - carbono[o]) * (costo[i] - costo[o])
            if cruz >= 0:
                H.pop()
            else:
                break
        H.append(i)
    return set(H)


def decision_a_posteriori(df=None, w=None, precio_carbono=None, pasos_w=21):
    """Elige un punto de la frontera YA CALCULADA (no se re-optimiza).
      - Suma ponderada normalizada: max  w·z_carbono + (1-w)·z_costo,
            z_carbono = (G - G_min)/(G_max - G_min),  z_costo = (C_max - C)/(C_max - C_min)  (1 = mejor en ambos)
      - Sensibilidad: qué punto se elige para w = 0, 0.05, ..., 1 (cada punto 'gana' un intervalo de pesos)
      - TOPSIS con los mismos pesos (cercanía relativa a la solución ideal), como contraste
      - Si se da precio_carbono p: max  p·carbono - costo  (peso económico, sin normalizar)
    Guarda decision_<modo>.csv y decision_sensibilidad_<modo>.csv."""
    import numpy as np
    w = cfg.PESO_CARBONO if w is None else w
    precio_carbono = cfg.PRECIO_CARBONO if precio_carbono is None else precio_carbono
    if df is None:
        ruta = os.path.join(cfg.CARPETA_RESULTADOS, 'frontera_combinada.csv')
        if not os.path.exists(ruta):
            print(f'No existe {ruta}: corre primero ambos barridos y combinar_fronteras().')
            return None
        df = pd.read_csv(ruta, encoding='utf-8-sig')
    df = df.sort_values('costo').reset_index(drop=True).copy()
    C, G = df['costo'].to_numpy(float), df['carbono'].to_numpy(float)
    rc, rg = max(C.max() - C.min(), 1e-9), max(G.max() - G.min(), 1e-9)
    df['z_costo'] = (C.max() - C) / rc
    df['z_carbono'] = (G - G.min()) / rg
    sop = _soportados(list(C), list(G))
    df['soportado'] = [i in sop for i in range(len(df))]

    # suma ponderada con el peso elegido
    df['score_ponderado'] = w * df['z_carbono'] + (1 - w) * df['z_costo']
    i_ws = int(df['score_ponderado'].idxmax())

    # sensibilidad del peso
    sens = []
    for wk in np.linspace(0, 1, pasos_w):
        s = wk * df['z_carbono'] + (1 - wk) * df['z_costo']
        k = int(s.idxmax())
        sens.append({'w_carbono': round(wk, 4), 'w_costo': round(1 - wk, 4), 'punto_elegido': df.loc[k, 'punto'],
                     'barrido': df.loc[k, 'barrido'], 'costo': df.loc[k, 'costo'], 'carbono': df.loc[k, 'carbono']})
    sens = pd.DataFrame(sens)

    # TOPSIS (2 criterios: carbono beneficio, costo a minimizar)
    rG, rC = G / np.linalg.norm(G), C / np.linalg.norm(C)
    vG, vC = w * rG, (1 - w) * rC
    d_mas = np.sqrt((vG - vG.max()) ** 2 + (vC - vC.min()) ** 2)
    d_menos = np.sqrt((vG - vG.min()) ** 2 + (vC - vC.max()) ** 2)
    df['topsis'] = d_menos / np.maximum(d_mas + d_menos, 1e-12)
    i_tp = int(df['topsis'].idxmax())

    elegidos = {'suma_ponderada': i_ws, 'topsis': i_tp}
    if precio_carbono is not None:
        df['valor_neto'] = precio_carbono * df['carbono'] - df['costo']
        elegidos['precio_carbono'] = int(df['valor_neto'].idxmax())

    df['elegido_por'] = ''
    for metodo, i in elegidos.items():
        df.loc[i, 'elegido_por'] = (df.loc[i, 'elegido_por'] + ' ' + metodo).strip()

    print(f'Suma ponderada (w_carbono = {w}, w_costo = {1 - w}): punto {df.loc[i_ws, "barrido"]} {df.loc[i_ws, "punto"]} '
          f'| costo {C[i_ws]:,.1f} | carbono {G[i_ws]:,.1f}')
    print(f'TOPSIS (mismos pesos): punto {df.loc[i_tp, "barrido"]} {df.loc[i_tp, "punto"]} '
          f'| costo {C[i_tp]:,.1f} | carbono {G[i_tp]:,.1f}')
    if precio_carbono is not None:
        k = elegidos['precio_carbono']
        print(f'Precio del carbono {precio_carbono:,.2f} $/unidad: punto {df.loc[k, "barrido"]} {df.loc[k, "punto"]} '
              f'| costo {C[k]:,.1f} | carbono {G[k]:,.1f}')
    print(f'Puntos soportados (los únicos que la suma ponderada puede elegir): {int(df["soportado"].sum())} de {len(df)}')
    tramos = sens.groupby((sens['punto_elegido'] + sens['barrido']).ne(
        (sens['punto_elegido'] + sens['barrido']).shift()).cumsum())
    print('Sensibilidad del peso del carbono:')
    for _, g in tramos:
        print(f'  w_carbono en [{g.w_carbono.min():.2f}, {g.w_carbono.max():.2f}] -> {g.barrido.iloc[0]} {g.punto_elegido.iloc[0]}')

    df.to_csv(os.path.join(cfg.CARPETA_RESULTADOS, f'decision_{cfg.MODO}.csv'), index=False, encoding='utf-8-sig')
    sens.to_csv(os.path.join(cfg.CARPETA_RESULTADOS, f'decision_sensibilidad_{cfg.MODO}.csv'), index=False, encoding='utf-8-sig')
    print(f'Guardado: decision_{cfg.MODO}.csv y decision_sensibilidad_{cfg.MODO}.csv')
    return df, sens
