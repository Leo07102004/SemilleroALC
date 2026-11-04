"""
Clasificación Bayesiana No Paramétrica mediante Ventanas de Parzen (Parzen-Rosenblatt)
======================================================================================

Este script implementa desde cero un clasificador basado en la estimación de densidad
por Ventanas de Parzen y lo evalúa sobre un DataFrame sintético multiclase no lineal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ==============================================================================
# 1. GENERACIÓN DEL DATAFRAME SINTÉTICO
# ==============================================================================
def generar_dataframe_sintetico(n_muestras: int = 450, semilla: int = 42) -> pd.DataFrame:
    """
    Genera un DataFrame sintético con 3 clases en un espacio 2D con estructuras
    no lineales y multimodales ideales para probar métodos no paramétricos.
    """
    rng = np.random.default_rng(semilla)
    n_por_clase = n_muestras // 3

    # Clase 0: Estructura bimodal (dos sub-clusters)
    n0_a = n_por_clase // 2
    n0_b = n_por_clase - n0_a
    c0_a = rng.multivariate_normal(mean=[-2.0, 1.5], cov=[[0.35, 0.1], [0.1, 0.35]], size=n0_a)
    c0_b = rng.multivariate_normal(mean=[1.5, -2.0], cov=[[0.40, -0.15], [-0.15, 0.40]], size=n0_b)
    X0 = np.vstack([c0_a, c0_b])

    # Clase 1: Arco semicircular (no lineal)
    angulos = rng.uniform(0.1 * np.pi, 0.9 * np.pi, size=n_por_clase)
    radios = rng.normal(loc=2.2, scale=0.28, size=n_por_clase)
    X1 = np.column_stack([
        radios * np.cos(angulos) + 0.5,
        radios * np.sin(angulos) - 0.2,
    ])

    # Clase 2: Núcleo central denso elíptico
    X2 = rng.multivariate_normal(mean=[0.2, 0.2], cov=[[0.45, 0.20], [0.20, 0.30]], size=n_por_clase)

    X = np.vstack([X0, X1, X2])
    y = np.array([0] * n_por_clase + [1] * n_por_clase + [2] * n_por_clase)
    nombres_clases = {0: "Clase 0 (Bimodal)", 1: "Clase 1 (Arco)", 2: "Clase 2 (Núcleo)"}

    df = pd.DataFrame({
        "x1": X[:, 0],
        "x2": X[:, 1],
        "clase": y,
        "etiqueta": [nombres_clases[c] for c in y],
    })

    # Mezclar filas aleatoriamente
    return df.sample(frac=1.0, random_state=semilla).reset_index(drop=True)


# ==============================================================================
# 2. IMPLEMENTACIÓN DEL CLASIFICADOR DE VENTANAS DE PARZEN
# ==============================================================================
class ParzenWindowClassifier:
    """
    Clasificador Bayesiano no paramétrico basado en Ventanas de Parzen-Rosenblatt.

    Estima la densidad condicional de cada clase como:
        p_hat(x | w_k) = (1 / (N_k * h^d)) * sum_{i=1}^{N_k} phi((x - x_i) / h)

    Y clasifica usando la regla de decisión de Bayes (MAP):
        w_hat(x) = argmax_k [ P(w_k) * p_hat(x | w_k) ]
    """

    def __init__(self, h: float = 0.5, kernel: str = "gaussiano") -> None:
        if h <= 0:
            raise ValueError("El ancho de ventana 'h' debe ser estrictamente positivo.")
        if kernel not in ("hipercubo", "gaussiano", "epanechnikov"):
            raise ValueError("Kernel no soportado. Usa 'hipercubo', 'gaussiano' o 'epanechnikov'.")

        self.h = float(h)
        self.kernel = kernel
        self.clases_: np.ndarray | None = None
        self.datos_por_clase_: dict[int, np.ndarray] = {}
        self.priors_: dict[int, float] = {}
        self.d_: int = 0

    def _funcion_ventana(self, u: np.ndarray) -> np.ndarray:
        """
        Evalúa la función ventana phi(u) donde u tiene forma (..., d).
        Garantiza que la integral sobre R^d sea 1.
        """
        d = u.shape[-1]

        if self.kernel == "hipercubo":
            # Ventana hipercúbica original de Parzen: 1 si |u_j| <= 0.5 para todo j, 0 en otro caso
            dentro = np.all(np.abs(u) <= 0.5, axis=-1)
            return dentro.astype(float)

        elif self.kernel == "gaussiano":
            # Ventana Gaussiana multivariada estándar
            norma_cuadrada = np.sum(u ** 2, axis=-1)
            constante = 1.0 / ((2.0 * np.pi) ** (d / 2.0))
            return constante * np.exp(-0.5 * norma_cuadrada)

        elif self.kernel == "epanechnikov":
            # Kernel parabólico de Epanechnikov (compacto y suave)
            norma_cuadrada = np.sum(u ** 2, axis=-1)
            # Constante de normalización para d dimensiones (volumen de bola unitaria)
            from scipy.special import gamma
            vol_bola = (np.pi ** (d / 2.0)) / gamma(d / 2.0 + 1.0)
            constante = ((d + 2.0) / (2.0 * vol_bola))
            return constante * np.maximum(0.0, 1.0 - norma_cuadrada)

        raise RuntimeError("Kernel inválido.")

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "ParzenWindowClassifier":
        """Almacena las muestras de entrenamiento por clase y calcula las probabilidades a priori."""
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)

        self.d_ = X_arr.shape[1]
        self.clases_ = np.unique(y_arr)
        n_total = len(y_arr)

        self.datos_por_clase_.clear()
        self.priors_.clear()

        for c in self.clases_:
            mascara = (y_arr == c)
            self.datos_por_clase_[int(c)] = X_arr[mascara]
            self.priors_[int(c)] = float(np.sum(mascara)) / n_total

        return self

    def estimar_densidad_clase(self, X: np.ndarray, clase: int) -> np.ndarray:
        """
        Calcula p_hat(x | w_k) para cada fila x de X respecto a una clase dada.
        """
        X_train_c = self.datos_por_clase_[int(clase)]
        n_k = len(X_train_c)
        volumen = self.h ** self.d_

        # Diferencias escaladas: u = (x - x_i) / h -> shape (n_eval, n_k, d)
        u = (X[:, np.newaxis, :] - X_train_c[np.newaxis, :, :]) / self.h
        pesos = self._funcion_ventana(u)  # shape (n_eval, n_k)

        densidad = np.sum(pesos, axis=1) / (n_k * volumen)
        return densidad

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Calcula las probabilidades a posteriori P(w_k | x) mediante el Teorema de Bayes.
        Si para una ventana compacta ninguna muestra cae dentro de la ventana,
        asigna las probabilidades a priori.
        """
        X_arr = np.asarray(X, dtype=float)
        n_muestras = X_arr.shape[0]
        n_clases = len(self.clases_)

        numeradores = np.zeros((n_muestras, n_clases), dtype=float)
        for idx, c in enumerate(self.clases_):
            verosimilitud = self.estimar_densidad_clase(X_arr, int(c))
            numeradores[:, idx] = verosimilitud * self.priors_[int(c)]

        evidencia = np.sum(numeradores, axis=1, keepdims=True)
        # Evitar división por cero en zonas sin soporte (p. ej. ventana hipercúbica pequeña)
        sin_soporte = (evidencia.squeeze(-1) == 0.0)
        evidencia_segura = np.where(evidencia == 0.0, 1.0, evidencia)

        posteriores = numeradores / evidencia_segura
        if np.any(sin_soporte):
            priors_vec = np.array([self.priors_[int(c)] for c in self.clases_])
            posteriores[sin_soporte] = priors_vec

        return posteriores

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predice la etiqueta de clase con máxima probabilidad a posteriori."""
        posteriores = self.predict_proba(X)
        indices_max = np.argmax(posteriores, axis=1)
        return self.clases_[indices_max]


# ==============================================================================
# 3. MÉTRICAS Y EVALUACIÓN
# ==============================================================================
def calcular_metricas(y_real: np.ndarray, y_pred: np.ndarray, clases: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Calcula exactitud (accuracy) y matriz de confusión en un DataFrame."""
    exactitud = float(np.mean(y_real == y_pred))
    matriz = np.zeros((len(clases), len(clases)), dtype=int)
    for r, p in zip(y_real, y_pred):
        matriz[int(r), int(p)] += 1

    df_conf = pd.DataFrame(
        matriz,
        index=[f"Real {c}" for c in clases],
        columns=[f"Pred {c}" for c in clases],
    )
    return exactitud, df_conf


# ==============================================================================
# 4. VISUALIZACIÓN DE FRONTERAS DE DECISIÓN Y DENSIDADES
# ==============================================================================
def graficar_resultados(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    ruta_salida: str = "resultados_parzen.png",
) -> None:
    """
    Genera una figura comparativa mostrando el efecto del ancho de ventana h
    y el tipo de función núcleo sobre la frontera de decisión y la densidad estimada.
    """
    configuraciones = [
        ("hipercubo", 0.65, "Ventana Hipercúbica (h = 0.65)"),
        ("gaussiano", 0.15, "Gaussiana (h = 0.15 - Sobreajuste)"),
        ("gaussiano", 0.45, "Gaussiana (h = 0.45 - Óptimo)"),
        ("gaussiano", 1.50, "Gaussiana (h = 1.50 - Subajuste)"),
    ]

    x_min, x_max = X_train[:, 0].min() - 1.0, X_train[:, 0].max() + 1.0
    y_min, y_max = X_train[:, 1].min() - 1.0, X_train[:, 1].max() + 1.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 140), np.linspace(y_min, y_max, 140))
    puntos_malla = np.c_[xx.ravel(), yy.ravel()]

    cmap_fondo = ListedColormap(["#ffcccc", "#cce5ff", "#d4edda"])
    colores_puntos = ["#d9534f", "#0275d8", "#28a745"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=130)
    axes = axes.ravel()

    for ax, (kernel, h, titulo) in zip(axes, configuraciones):
        clf = ParzenWindowClassifier(h=h, kernel=kernel).fit(X_train, y_train)
        pred_test = clf.predict(X_test)
        acc = np.mean(pred_test == y_test)

        # Predicción en la malla y densidad total p(x)
        Z = clf.predict(puntos_malla).reshape(xx.shape)
        densidad_total = sum(
            clf.estimar_densidad_clase(puntos_malla, int(c)) * clf.priors_[int(c)]
            for c in clf.clases_
        ).reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.45, cmap=cmap_fondo, levels=[-0.5, 0.5, 1.5, 2.5])
        ax.contour(xx, yy, densidad_total, levels=6, colors="black", alpha=0.25, linewidths=0.8)

        for c in clf.clases_:
            mask_tr = (y_train == c)
            mask_te = (y_test == c)
            ax.scatter(
                X_train[mask_tr, 0], X_train[mask_tr, 1],
                c=colores_puntos[int(c)], label=f"Clase {c} (Train)",
                edgecolor="k", s=28, alpha=0.75, marker="o"
            )
            ax.scatter(
                X_test[mask_te, 0], X_test[mask_te, 1],
                c=colores_puntos[int(c)],
                edgecolor="k", s=52, alpha=0.95, marker="^"
            )

        ax.set_title(f"{titulo}\nExactitud Test: {acc * 100:.2f}%", fontsize=11, fontweight="bold")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.grid(True, linestyle="--", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, fontsize=10)
    plt.suptitle(
        "Clasificación con Ventanas de Parzen: Fronteras de Decisión y Curvas de Iso-Densidad",
        fontsize=14, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(ruta_salida)
    plt.close(fig)


# ==============================================================================
# 5. FLUJO PRINCIPAL
# ==============================================================================
def main() -> None:
    print("=" * 72)
    print("  CLASIFICACIÓN BAYESIANA MEDIANTE VENTANAS DE PARZEN-ROSENBLATT")
    print("=" * 72)

    # 1. Crear y guardar DataFrame sintético
    df = generar_dataframe_sintetico(n_muestras=450, semilla=42)
    df.to_csv("datos_sinteticos.csv", index=False)
    print("\n[1] Primeras 5 filas del DataFrame sintético ('datos_sinteticos.csv'):")
    print(df.head())

    print("\n[2] Resumen estadístico por clase:")
    print(df.groupby("etiqueta")[["x1", "x2"]].describe().round(3))

    # 2. División Entrenamiento (75%) / Prueba (25%)
    n_train = int(0.75 * len(df))
    df_train = df.iloc[:n_train]
    df_test = df.iloc[n_train:]

    X_train = df_train[["x1", "x2"]].to_numpy()
    y_train = df_train["clase"].to_numpy()
    X_test = df_test[["x1", "x2"]].to_numpy()
    y_test = df_test["clase"].to_numpy()

    # 3. Comparación de hiperparámetros (kernel y ancho de ventana h)
    print("\n[3] Evaluación comparativa de Kernels y Anchos de Ventana (h):")
    experimentos = [
        ("hipercubo", 0.35),
        ("hipercubo", 0.65),
        ("hipercubo", 1.20),
        ("epanechnikov", 0.65),
        ("gaussiano", 0.15),
        ("gaussiano", 0.45),
        ("gaussiano", 1.50),
    ]

    filas_resumen = []
    for kernel, h in experimentos:
        clf = ParzenWindowClassifier(h=h, kernel=kernel).fit(X_train, y_train)
        acc_train = float(np.mean(clf.predict(X_train) == y_train))
        acc_test = float(np.mean(clf.predict(X_test) == y_test))
        filas_resumen.append({
            "Kernel": kernel.capitalize(),
            "Ancho_h": h,
            "Acc_Train (%)": round(acc_train * 100, 2),
            "Acc_Test (%)": round(acc_test * 100, 2),
        })

    df_resultados = pd.DataFrame(filas_resumen)
    print(df_resultados.to_string(index=False))

    # 4. Detalle del mejor modelo (Gaussiano con h = 0.45)
    mejor_modelo = ParzenWindowClassifier(h=0.45, kernel="gaussiano").fit(X_train, y_train)
    y_pred = mejor_modelo.predict(X_test)
    acc, df_confusion = calcular_metricas(y_test, y_pred, mejor_modelo.clases_)

    print(f"\n[4] Matriz de Confusión en Test (Kernel Gaussiano, h = 0.45 | Exactitud = {acc * 100:.2f}%):")
    print(df_confusion)

    # 5. Generar visualización
    graficar_resultados(X_train, y_train, X_test, y_test, ruta_salida="resultados_parzen.png")
    print("\n[5] Gráfica guardada exitosamente en 'resultados_parzen.png'.")


if __name__ == "__main__":
    main()
