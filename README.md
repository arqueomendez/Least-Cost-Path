# Least-Cost-Path

## Descripción
Least-Cost-Path (LCP) es un software de análisis espacial avanzado para el cálculo de rutas de menor coste sobre superficies raster, utilizando algoritmos jerárquicos y optimizados. Permite encontrar rutas óptimas entre puntos o entre un punto y todos los demás, considerando superficies de fricción y restricciones espaciales.

## Características principales
- Implementa el algoritmo A* optimizado con Numba para alto rendimiento.
- Soporta análisis jerárquico en dos fases: primero en baja resolución para encontrar un corredor estratégico y luego en alta resolución para el detalle final.
- **NUEVO**: Permite restringir el área de búsqueda a un polígono vectorial (shapefile) para evitar rutas no deseadas y mejorar la precisión.
- Permite calcular rutas entre dos puntos o desde un punto origen a todos los destinos definidos en un shapefile.
- Exporta los resultados como shapefiles compatibles con SIG.
- Incluye scripts de visualización para comparar rutas y analizar resultados.

## Estructura del proyecto
- `LCP.ipynb`: Notebook principal con todo el flujo de análisis, desde la configuración hasta la visualización.
- `pyproject.toml`: Definición de dependencias y metadatos del proyecto.
- `uv.lock`: Detalles de versiones de dependencias instaladas.
- `README.md`: Este archivo.


- numpy, rasterio, fiona, geopandas, shapely, scikit-image, matplotlib, matplotlib-scalebar, tqdm, numba, pandas, jupyterlab

## Instalación y uso con UV y Jupyter Lab

1. Instala [uv](https://github.com/astral-sh/uv) si no lo tienes:

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

2. Instala todas las dependencias del proyecto usando `uv add`:
   
   ```sh
   uv add -r requirements.txt
   # o si usas pyproject.toml
   uv add -r pyproject.toml
   ```

3. Inicia Jupyter Lab:
   
   ```sh
   uv add jupyterlab  # Si no está instalado
   jupyter lab
   ```

4. Abre y ejecuta el notebook `LCP.ipynb`.

## Uso básico
1. Prepara los archivos de entrada:
	- Un raster de coste (`final_total_cost.tif`).
	- Shapefiles de puntos de inicio y fin, o un shapefile con todos los puntos.
	- **Opcional**: Un shapefile con un polígono que defina el área de búsqueda permitida.
2. Abre y ejecuta el notebook `LCP.ipynb` en JupyterLab.
3. Ajusta las rutas y parámetros en las primeras celdas:
    - `BASE_PATH`: La carpeta principal de tu proyecto.
    - `COST_RASTER_PATH`: La ruta a tu ráster de coste.
    - `START_SHAPEFILE_PATH` / `END_SHAPEFILE_PATH`: Rutas a los puntos de inicio/fin.
    - `MASK_SHAPEFILE_PATH`: **Importante**, establece aquí la ruta a tu shapefile de máscara o déjalo como `None` si no quieres usarlo.
4. Ejecuta las celdas para calcular rutas y exportar resultados.
5. Usa las funciones de visualización para analizar y comparar rutas.

## Ejemplo de flujo de trabajo
1. Define la carpeta base y los archivos de entrada en la celda de configuración.
2. Ejecuta la celda principal para calcular la ruta entre dos puntos o de uno a todos.
3. Los resultados se guardan en una carpeta con marca de tiempo.
4. Ejecuta la celda de visualización para generar mapas comparativos.

## Método de trabajo (según LCP.ipynb)

El flujo de trabajo implementado en el notebook `LCP.ipynb` sigue una metodología robusta y reproducible para el cálculo de rutas de menor coste:

1. **Configuración y validación de insumos**
   - Se definen rutas y parámetros clave (carpetas, archivos raster y vectoriales, IDs de puntos, factores de downsampling, etc.).
   - Se valida que todos los puntos de entrada estén dentro del área permitida por la máscara poligonal. Si algún punto está fuera, el proceso se cancela y se registra el error.

2. **Gestión de sesiones y reanudación automática**
   - El sistema crea una carpeta de sesión con marca de tiempo para cada ejecución.
   - Si existe una sesión previa incompleta, el proceso la detecta y reanuda automáticamente desde el último punto pendiente.
   - El registro maestro (`registro_maestro_procesamiento.log`) documenta todo el proceso y los errores.

3. **Cálculo jerárquico de rutas (A* optimizado)**
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


## Licencia y atribución
Este software se distribuye bajo una licencia de atribución, uso no comercial y compartida:

- Puedes modificar y compartir el código libremente, siempre y cuando:
	- No lo utilices con fines comerciales ni en productos o servicios comerciales.
	- Mantengas esta licencia y la sección de atribución en cualquier copia o derivado.
	- Cites explícitamente el siguiente texto en cualquier uso, publicación o derivado:

		> "Least-Cost-Path (LCP) desarrollado por Víctor Méndez."

- El uso en investigación, docencia y proyectos personales está permitido.
- Para cualquier uso comercial, se debe solicitar autorización expresa al autor.

## Autoría
Desarrollado por Víctor Méndez, con asistencia de Gemini 2.5 Pro.
