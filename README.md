# Least-Cost-Path

[![DOI](https://zenodo.org/badge/1051193732.svg)](https://doi.org/10.5281/zenodo.17172977)

## Descripción
Least-Cost-Path (LCP) es un software de análisis espacial avanzado para el cálculo de rutas de menor coste sobre superficies raster, utilizando algoritmos jerárquicos y optimizados. Permite encontrar rutas óptimas entre puntos o entre un punto y todos los demás, considerando superficies de fricción y restricciones espaciales. Least-Cost-Path ha sido desarrollado para ser utilizado en arqueología, con la finalidad de realizar análisis de movimientos en diversos tipos de espacios. La superficie de raster a partir del cual se realiza el cálculo debe ser considerado un modelo teórico que contiene el coste o fricción para desplazarse entre los pixeles. 

## Características principales
- Implementa el algoritmo A* optimizado con Numba para alto rendimiento.
- Soporta análisis jerárquico en dos fases: primero en baja resolución para encontrar un corredor estratégico y luego en alta resolución para el detalle final.
- Permite restringir el área de búsqueda a un polígono vectorial (shapefile) para evitar rutas no deseadas y mejorar la precisión.
- Permite calcular rutas entre dos puntos o desde un punto origen a todos los destinos definidos en un shapefile.
- Exporta los resultados como shapefiles compatibles con SIG.
- Incluye scripts de visualización para comparar rutas y analizar resultados.
- Es posible correr el software tanto directamente en un ejecutable directo en Python como mediante Jupyter Notebook.

## Estructura y estado actual del proyecto

### Carpetas principales
- **`data/`**: Archivos de entrada de prueba (raster de coste, shapefiles de puntos y máscara poligonal).
- **`output/`**: Resultados de cada sesión, organizados por fecha/hora, con shapefiles de rutas calculadas.
- **`lcp/`**: Módulo principal con la lógica del proyecto:
   - `data_loader.py`: Carga raster, puntos y máscara.
   - `processing.py`: Procesamiento raster, downsampling, creación de máscaras y corredores.
   - `pathfinder.py`: Algoritmo A* optimizado con Numba.
   - `utils.py`: Utilidades para guardar rutas y manejo de geometrías.
   - `__init__.py`: Inicialización del paquete.

### Archivos principales
- **`lcp.py`**: Script ejecutable que orquesta el análisis completo fuera de Jupyter.
- **`LCP_N2N.ipynb`**: Notebooks para ejecutar el análisis de todos a todos los puntos, visualizar resultados y probar variantes.
- **`LCP_base.ipynb`**: Notebooks para ejecutar el análisis de un punto a todos, visualizar resultados y probar variantes.
- **`LCP_VSH.py`**: Script alternativo para flujos personalizados o pruebas.
- **`pyproject.toml`**: Configuración y dependencias del proyecto.
- **`uv.lock`**: Bloqueo de versiones de dependencias.
- **`LICENSE`**: Licencia de uso (CC BY-NC 4.0).
- **`README.md`**: Este archivo.

### Función de cada archivo
- **Notebooks (`*.ipynb`)**: Ejecución paso a paso, visualización y comparación de rutas.
- **Scripts (`*.py`)**: Automatización del flujo, cálculo y manejo de datos.
- **Carpetas `data/` y `output/`**: Entrada y salida del sistema.
- **Módulos en `lcp/`**: Lógica separada en carga de datos, procesamiento, cálculo de rutas y utilidades.

### Dependencias utilizadas
El proyecto utiliza las siguientes dependencias principales (ver `pyproject.toml`):
- `fiona`
- `geopandas`
- `jupyterlab`
- `matplotlib-scalebar`
- `numba`
- `numpy`
- `pandas`
- `rasterio`
- `scikit-image`
- `shapely`
- `tqdm`

Estas cubren la carga y manejo de datos espaciales, procesamiento raster, cálculo de rutas, optimización, visualización y ejecución en notebooks.

## Instalación y uso con UV y Jupyter Lab

1. Instala [uv](https://github.com/astral-sh/uv) si no lo tienes:
	
	a. Con Pypy:

   - **Windows**:
     ```sh
     pip install uv
     ```
   - **macOS**:
     ```sh
     brew install uv
     # o
     pip install uv
     ```
   - **Linux**:
     ```sh
     pipx install uv
     # o
     pip install uv
     ```
	
	b. Standalone

   - **Windows**:
     ```sh
     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
     ```
   - **macOS y Linux **:
     ```sh
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```

3. Clona el repositorio usando:
   
   ```sh
   git clone https://github.com/arqueomendez/Least-Cost-Path.git
   cd Least-Cost-Path
   ```

4. Instala todas las dependencias del proyecto usando:
   
   - **Con uv**:
   ```sh
   uv sync
   ```

   - **Con pip**:
   ```sh
   pip install -r requirements.txt
   ```

5. Inicia Jupyter Lab:
   
   ```sh
   uv run jupyter lab
   ```

6. Abre y ejecuta el notebook `LCP.ipynb`.

## Uso básico
1. Prepara los archivos de entrada:
	- Un raster de coste (`cost.tif`).
	- Shapefiles con todos los puntos (`points.shp`).
	- **Opcional**: Un shapefile con un polígono que defina el área de búsqueda permitida (`area-mask.shp`).
2. Abre y ejecuta el notebook `LCP_base.ipynb` en JupyterLab.
3. Ajusta las rutas y parámetros en las primeras celdas:
    - `BASE_PROCESSING_FOLDER`: La carpeta principal de tu proyecto.
    - `POINTS_SUBFOLDER`: Rutas a los puntos de inicio/fin.
    - `COST_RASTER_PATH`: La ruta a tu ráster de coste.
    - `ALL_POINTS_SHAPEFILE`: Shapefiles con todos los puntos (`points.shp`).
    - `MASK_SHAPEFILE_PATH`: **Importante**, establece aquí la ruta a tu shapefile de máscara o déjalo como `None` si no quieres usarlo.
4. Ejecuta las celdas para calcular rutas y exportar resultados.
5. Usa las funciones de visualización para analizar y comparar rutas.

## Ejemplo de flujo de trabajo
1. Define la carpeta base y los archivos de entrada en la celda de configuración.
2. Ejecuta la celda principal para calcular la ruta entre dos puntos o de uno a todos.
3. Los resultados se guardan en una carpeta con marca de tiempo.
4. Ejecuta la celda de visualización para generar mapas comparativos.

## Método de trabajo (según LCP_base.ipynb)

El flujo de trabajo implementado en el notebook `LCP_base.ipynb` sigue una metodología robusta y reproducible para el cálculo de rutas de menor coste:

1. **Configuración y validación de insumos**
   - Se definen rutas y parámetros clave (carpetas, archivos raster y vectoriales, IDs de puntos, factores de downsampling, etc.).
   - Se valida que todos los puntos de entrada estén dentro del área permitida por la máscara poligonal. Si algún punto está fuera, el proceso se cancela y se registra el error.

2. **Gestión de sesiones y reanudación automática**
   - El sistema crea una carpeta de sesión con marca de tiempo para cada ejecución.
   - Si existe una sesión previa incompleta, el proceso la detecta y reanuda automáticamente desde el último punto pendiente.
   - El registro maestro (`registro_maestro_procesamiento.log`) documenta todo el proceso y los errores.

3. **Cálculo jerárquico de rutas (A* optimizado)**
   - Si la máscara está activada, se realiza un cálculo para verificar que todos los puntos se encuentren dentro del espacio. Si no es así, el análisis se detiene.
   - Para cada destino, se realiza primero una búsqueda en baja resolución (downsampling) usando varios factores. Esto permite encontrar un corredor estratégico de menor coste.
   - Si la búsqueda en baja resolución tiene éxito, se genera un corredor de búsqueda en alta resolución alrededor de la ruta preliminar.
   - Se realiza la búsqueda final en alta resolución, restringida al corredor. Si falla, se aplica un "plan B" usando toda la máscara rasterizada.
   - Todo el proceso incluye logging detallado y reportes de progreso integrados con tqdm.

4. **Exportación y visualización de resultados**
   - Las rutas calculadas se exportan como shapefiles, tanto de la fase preliminar como de la final.
   - El notebook incluye funciones para visualizar y comparar rutas exportadas.

5. **Robustez y reproducibilidad**
   - El código maneja errores de insumos, proyecciones y rutas vacías.
   - El uso de máscaras vectoriales y validación previa garantiza que los resultados sean espacialmente coherentes.
   - El sistema de logging y reanudación automática permite ejecutar análisis largos sin perder avances.

## Notas técnicas
- El código está optimizado para grandes superficies raster y rutas largas.
- Se recomienda usar rutas relativas para facilitar la portabilidad del proyecto.
- El notebook incluye manejo de errores para archivos de entrada y proyecciones.

## Contribución

¡Contribuciones bienvenidas! Puedes ayudar en:
- Optimización de rendimiento
- Documentación y tutoriales

## Licencia

Este proyecto está licenciado bajo **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.

- Puedes usar, modificar y compartir el software libremente **solo para fines no comerciales**.
- Es obligatorio citar el proyecto y a los autores originales.
- Para uso comercial, contacta a los autores.

Ver el archivo LICENSE para detalles completos.

## Cita y Autores

If you use this software in your research, please cite it as follows:

```bibtex
@software{Mendez_Least_Cost_Path,
  author = {Méndez, Víctor},
  title = {{Least-Cost-Path: A Python tool for Least Cost Path analysis in archaeology}},
  url = {https://github.com/arqueomendez/Least-Cost-Path},
  year = {2025}
}

## Soporte y Comunidad

- Issues: Reporta bugs y solicita mejoras en GitHub Issues
- Documentación: Consulta la documentación técnica en `docs/`
- Comunidad: Únete a foros y discusiones científicas