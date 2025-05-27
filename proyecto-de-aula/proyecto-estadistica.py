import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

def interpretar_asimetria_curtosis(asimetria, curtosis):
    texto = "📈 Interpretación de Forma de la Distribución\n\n"

    # Interpretar asimetría (skewness)
    if abs(asimetria) < 0.5:
        texto += "La distribución es aproximadamente simétrica.\n"
    elif asimetria > 0:
        texto += "La distribución está sesgada a la derecha (asimetría positiva): cola más larga hacia valores altos.\n"
    else:
        texto += "La distribución está sesgada a la izquierda (asimetría negativa): cola más larga hacia valores bajos.\n"

    # Interpretar curtosis
    # Curtosis > 0 leptocúrtica (colas pesadas), curtosis < 0 platicúrtica (colas ligeras), cerca de 0 mesocúrtica (normal)
    if curtosis > 1:
        texto += "La distribución es leptocúrtica: tiene colas pesadas y pico agudo, más valores extremos.\n"
    elif curtosis < -1:
        texto += "La distribución es platicúrtica: tiene colas ligeras y pico plano, menos valores extremos.\n"
    else:
        texto += "La distribución es mesocúrtica: similar a una distribución normal en cuanto a la forma del pico y colas.\n"

    return texto


def interpretar_correlacion(var1, var2, r, p):
    texto = f"Interpretación\n"
    texto += f"📉 Coeficiente: {r:.4f}\n"
                
    # Dirección
    if r < 0:
        texto += f"Signo negativo: indica que a mayor valor de {var1}, menor valor de {var2}.\n"
    elif r > 0:
        texto += f"Signo positivo: indica que a mayor valor de {var1}, mayor valor de {var2}.\n"
    else:
        texto += "Coeficiente cero: no hay relación lineal entre las variables.\n"
                
    # Fuerza (magnitud)
    abs_r = abs(r)
    if abs_r < 0.1:
        texto += "Magnitud muy baja: la relación es casi nula o insignificante.\n"
    elif abs_r < 0.3:
        texto += "Magnitud baja: la relación es débil.\n"
    elif abs_r < 0.5:
        texto += "Magnitud moderada: la relación es moderada.\n"
    elif abs_r < 0.7:
        texto += "Magnitud alta: la relación es fuerte.\n"
    else:
        texto += "Magnitud muy alta: la relación es muy fuerte.\n"
                
    texto += f"O sea, {var1} y {var2} {'tienen una relación leve' if abs_r < 0.3 else 'están relacionadas'}.\n\n"

    texto += f"📊 p-valor: {p:.4f}\n"
    if p < 0.05:
        texto += "Como p < 0.05, esta correlación es estadísticamente significativa.\n"
        texto += "Hay evidencia suficiente para afirmar que existe una relación (aunque puede ser débil) entre ambas variables.\n\n"
    else:
        texto += "Como p ≥ 0.05, no hay evidencia suficiente para afirmar que la correlación sea significativa.\n\n"

    texto += "📌 Conclusión:\n"
    texto += (f"Se encontró una correlación {'negativa' if r < 0 else 'positiva'} "
                f"de fuerza {'baja' if abs_r < 0.3 else 'moderada' if abs_r < 0.5 else 'alta' if abs_r < 0.7 else 'muy alta'} "
                f"(r = {r:.4f}) entre {var1} y {var2}.\n")
    texto += f"El análisis arrojó un p-valor de {p:.4f}, "
    texto += ("lo que indica que esta correlación es estadísticamente significativa.\n" if p < 0.05 else "lo que indica que esta correlación no es estadísticamente significativa.\n")
                
    texto += ("Esto sugiere que, en este conjunto de datos, los valores de ambas variables están relacionados "
                        "de manera que a medida que una aumenta, la otra tiende a "
                        f"{'disminuir' if r < 0 else 'aumentar'}, aunque la fuerza de esta relación puede ser limitada.\n")
    return texto

st.set_page_config(page_title="Análisis Estadístico", layout="wide")
st.title("Análisis Estadístico")

# Subida de archivo
uploaded_file = st.file_uploader("Cargar archivo CSV o Excel", type=["csv", "xlsx"])

if uploaded_file:
    # Leer archivo
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Vista previa de los datos")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    selected_col = st.selectbox("Selecciona una columna numérica para análisis", numeric_cols)

    if selected_col:
        data = df[selected_col].dropna()

        st.subheader("Medidas de tendencia central")
        st.write(f"Media: {data.mean():.4f}")
        st.write(f"Mediana: {data.median():.4f}")
        mode_val = data.mode().values
        st.write(f"Moda: {mode_val[0]:.4f}" if len(mode_val) > 0 else "Moda: No disponible")

        st.subheader("Medidas de dispersión")
        st.write(f"Rango: {data.max() - data.min():.4f}")
        st.write(f"Desviación estándar: {data.std():.4f}")
        st.write(f"Coeficiente de variación: {(data.std() / data.mean()) * 100:.2f}%")

        st.subheader("Medidas de forma")
        asimetria = stats.skew(data)
        curtosis = stats.kurtosis(data)
        st.write(f"Asimetría (Skewness): {asimetria:.4f}")
        st.write(f"Curtosis: {curtosis:.4f}")

        # Mostrar interpretación automática
        interpretacion = interpretar_asimetria_curtosis(asimetria, curtosis)
        st.text_area("Interpretación de asimetría y curtosis", interpretacion, height=150)


        st.subheader("Visualizaciones")
        col1, col2 = st.columns(2)

        with col1:
            st.write("Histograma")
            fig, ax = plt.subplots()
            sns.histplot(data, kde=True, ax=ax)
            st.pyplot(fig)

        with col2:
            st.write("Diagrama de Caja")
            fig2, ax2 = plt.subplots()
            sns.boxplot(x=data, ax=ax2)
            st.pyplot(fig2)
        
        st.subheader("Diagrama de dispersión")

        other_cols = [col for col in df.columns if col != selected_col]
        second_col = st.selectbox("Selecciona otra columna para comparar", other_cols)


        if second_col:
            common_data = df[[selected_col, second_col]].dropna()
            common_data = df[[selected_col, second_col]].dropna()

            fig_scatter, ax_scatter = plt.subplots()

            if common_data[second_col].dtype == 'object' or common_data[second_col].dtype.name == 'category':
                sns.stripplot(x=second_col, y=selected_col, data=common_data, ax=ax_scatter, jitter=True)
                ax_scatter.set_title(f"Diagrama de dispersión (categoría vs numérica): {second_col} vs {selected_col}")
            elif common_data[selected_col].dtype == 'object' or common_data[selected_col].dtype.name == 'category':
                sns.stripplot(x=selected_col, y=second_col, data=common_data, ax=ax_scatter, jitter=True)
                ax_scatter.set_title(f"Diagrama de dispersión (categoría vs numérica): {selected_col} vs {second_col}")
            else:
                sns.scatterplot(x=selected_col, y=second_col, data=common_data, ax=ax_scatter, color="blue")
                ax_scatter.set_title(f"Diagrama de dispersión entre {selected_col} y {second_col}")

            st.pyplot(fig_scatter)

            # Correlación de Pearson
            corr_coef, p_val = stats.pearsonr(common_data[selected_col], common_data[second_col])
            st.write(f"Coeficiente de correlación de Pearson: {corr_coef:.4f}")
            st.write(f"p-valor: {p_val:.4f}")

            if p_val < 0.05:
                st.write("✅ Existe una correlación estadísticamente significativa.")
            else:
                st.write("ℹ️ No se encuentra una correlación estadísticamente significativa.")

            analisis_texto = interpretar_correlacion(selected_col, second_col, corr_coef, p_val)
            st.text_area("Análisis de la correlación", analisis_texto, height=250)


        st.subheader("Intervalo de confianza")
        confidence_level = st.selectbox("Selecciona el nivel de confianza", options=[0.90, 0.95, 0.99], index=1)
        alpha = 1 - confidence_level

        # Intervalo de confianza para la media
        ci_low, ci_high = stats.t.interval(confidence=confidence_level, df=len(data)-1, loc=data.mean(), scale=stats.sem(data))
        st.write(f"Intervalo de confianza para la media al {int(confidence_level*100)}%: ({ci_low:.4f}, {ci_high:.4f})")

        # Intervalo de confianza para la desviación estándar (basado en chi-cuadrado)
        n = len(data)
        sample_var = np.var(data, ddof=1)
        chi2_lower = stats.chi2.ppf(alpha / 2, df=n - 1)
        chi2_upper = stats.chi2.ppf(1 - alpha / 2, df=n - 1)
        ci_std_low = np.sqrt((n - 1) * sample_var / chi2_upper)
        ci_std_high = np.sqrt((n - 1) * sample_var / chi2_lower)
        st.write(f"Intervalo de confianza para la desviación estándar al {int(confidence_level*100)}%: ({ci_std_low:.4f}, {ci_std_high:.4f})")

        st.subheader("Prueba de hipótesis")
        mu_0 = st.number_input("Hipótesis nula: media =", value=float(data.mean()))
        t_stat, p_value = stats.ttest_1samp(data, mu_0)
        st.write(f"Estadístico t: {t_stat:.4f}")
        st.write(f"p-valor: {p_value:.4f}")

        if p_value < 1 - confidence_level:
            st.write(f"👉 Se rechaza la hipótesis nula al nivel de significancia del {int((1 - confidence_level)*100)}%.")
        else:
            st.write(f"✅ No se rechaza la hipótesis nula al nivel de significancia del {int((1 - confidence_level)*100)}%.")

        st.subheader("Análisis de distribución (Chi-cuadrado)")

        dist_option = st.selectbox(
            "Selecciona una distribución para ajustar",
            ["normal", "exponencial", "uniforme", "gamma", "poisson", "binomial"]
        )

        # Número de clases para frecuencias
        bins = st.slider("Número de clases", min_value=4, max_value=20, value=8)

        # Discretizar datos
        hist, bin_edges = np.histogram(data, bins=bins)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        if dist_option == "normal":
            params = stats.norm.fit(data)
            expected_freq = len(data) * (stats.norm.cdf(bin_edges[1:], *params) - stats.norm.cdf(bin_edges[:-1], *params))
        elif dist_option == "exponencial":
            params = stats.expon.fit(data)
            expected_freq = len(data) * (stats.expon.cdf(bin_edges[1:], *params) - stats.expon.cdf(bin_edges[:-1], *params))
        elif dist_option == "uniforme":
            params = stats.uniform.fit(data)
            expected_freq = len(data) * (stats.uniform.cdf(bin_edges[1:], *params) - stats.uniform.cdf(bin_edges[:-1], *params))
        elif dist_option == "gamma":
            params = stats.gamma.fit(data)
            expected_freq = len(data) * (stats.gamma.cdf(bin_edges[1:], *params) - stats.gamma.cdf(bin_edges[:-1], *params))

        elif dist_option == "poisson":
            # Solo válido para datos enteros no negativos
            if np.any(data < 0) or not np.all(np.floor(data) == data):
                st.warning("❗ La distribución Poisson solo es válida para datos enteros no negativos.")
                expected_freq = None
            else:
                mu = data.mean()
                values, counts = np.unique(data, return_counts=True)
                max_val = int(data.max())
                expected_freq = stats.poisson.pmf(np.arange(max_val + 1), mu) * len(data)
                # Truncar para coincidir con valores observados
                hist = np.array([np.sum(data == k) for k in range(max_val + 1)])
                bin_centers = np.arange(max_val + 1)

        elif dist_option == "binomial":
            # Solo válido para datos enteros y n definido
            if np.any(data < 0) or not np.all(np.floor(data) == data):
                st.warning("❗ La distribución Binomial solo es válida para datos enteros no negativos.")
                expected_freq = None
            else:
                n = int(data.max())
                p = data.mean() / n if n > 0 else 0.5
                values = np.arange(n + 1)
                expected_freq = stats.binom.pmf(values, n, p) * len(data)
                hist = np.array([np.sum(data == k) for k in values])
                bin_centers = values

        # Función para agrupar clases con frecuencias esperadas < 5
        def agrupar_bins(hist, expected):
            new_hist = []
            new_expected = []
            i = 0
            while i < len(expected):
                obs_sum = hist[i]
                exp_sum = expected[i]
                j = i
                while exp_sum < 5 and j + 1 < len(expected):
                    j += 1
                    obs_sum += hist[j]
                    exp_sum += expected[j]
                new_hist.append(obs_sum)
                new_expected.append(exp_sum)
                i = j + 1
            return np.array(new_hist), np.array(new_expected)

        if expected_freq is not None:
            # Agrupar clases si es necesario
            hist_grouped, expected_grouped = agrupar_bins(hist, expected_freq)

            # Solo hacer prueba si las longitudes coinciden y son válidas
            if len(hist_grouped) == len(expected_grouped) and len(hist_grouped) > 1:
                # Ajustar suma total de esperadas para que coincida con las observadas
                expected_grouped = expected_grouped * (hist_grouped.sum() / expected_grouped.sum())

                chi2_stat, p_val = stats.chisquare(f_obs=hist_grouped, f_exp=expected_grouped)

                st.write(f"**Prueba de Chi-cuadrado para distribución {dist_option.title()}**")
                st.write(f"Estadístico χ²: {chi2_stat:.4f}")
                st.write(f"p-valor: {p_val:.4f}")

                if p_val < 0.05:
                    st.write(f"❌ Se rechaza la hipótesis nula: los datos **no** siguen una distribución {dist_option.title()}.")
                else:
                    st.write(f"✅ No se rechaza la hipótesis nula: los datos **podrían** seguir una distribución {dist_option.title()}.")
                
                # Gráfico
                fig4, ax4 = plt.subplots(figsize=(8, 4))
                ax4.bar(range(len(hist_grouped)), hist_grouped, width=0.4, alpha=0.6, label="Observado", align='center')
                ax4.plot(range(len(expected_grouped)), expected_grouped, 'r--o', label="Esperado")
                ax4.set_title(f"Comparación Observado vs. Esperado ({dist_option.title()})")
                ax4.set_xlabel("Grupos")
                ax4.set_ylabel("Frecuencia")
                ax4.legend()
                st.pyplot(fig4)
            else:
                st.warning("❗ No se pudo realizar la prueba de Chi-cuadrado debido a grupos insuficientes o mal definidos.")

