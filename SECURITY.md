# Seguridad, privacidad y buenas prácticas

## Medidas incorporadas en la versión 2

- **Uso local por defecto:** Streamlit escucha en `127.0.0.1`; no se publica automáticamente en la red.
- **Límite de carga:** máximo 20 MiB por CSV, 200.000 filas y 200 columnas.
- **Validación de CSV:** comprobación de archivo vacío, cabeceras duplicadas, cabeceras sin nombre, filas mal formadas y límites estructurales.
- **Mensajes de error prudentes:** la interfaz no muestra directamente excepciones de librerías que pudieran revelar detalles internos o datos de entrada.
- **Exportaciones protegidas:** el texto que empieza con `=`, `+`, `-` o `@` se neutraliza en las exportaciones CSV y Excel para reducir el riesgo de inyección de fórmulas al abrirlas en hojas de cálculo.
- **Sin envío externo implementado:** el código no incorpora llamadas de red para transmitir los CSV. Los paquetes de Python se descargan al instalarlos y son una actividad aparte.
- **Estadísticas de uso del navegador desactivadas** en la configuración local de Streamlit.
- **Pruebas automatizadas** para lectura, cabeceras duplicadas, límite de tamaño, exportación segura y flujo analítico.

## Antes de usar datos reales

1. Usa únicamente las columnas necesarias; elimina nombres, correos, teléfonos y otros identificadores directos si no son imprescindibles. El ID que se mapea como identificador se excluye de los predictores, pero puede aparecer en las tablas/exportaciones para poder reconocer registros.
2. Guarda el proyecto y los CSV en un disco con permisos adecuados y cifrado si corresponde. No subas datos reales a repositorios públicos ni compartas capturas con información de clientes.
3. No expongas el servidor a Internet ni a una red compartida sin una arquitectura de despliegue con autenticación, autorización, TLS, gestión de secretos y revisión de seguridad. La aplicación no incorpora cuentas de usuario ni control de acceso.
4. Instala dependencias en un entorno virtual y revisa/actualiza versiones con un proceso controlado. No ejecutes el proyecto con privilegios de administrador.
5. Verifica el origen, la calidad, las clases objetivo y los resultados del modelo antes de tomar decisiones. Un nivel de riesgo no es una certeza ni una recomendación automática para contactar, excluir o penalizar clientes.
6. Define un periodo de conservación y elimina los archivos temporales/exportaciones cuando ya no sean necesarios.

## RGPD y límites

Estas medidas son buenas prácticas técnicas, **no una certificación de cumplimiento del RGPD** ni una evaluación jurídica. Antes de tratar datos personales reales, la organización debe determinar la base jurídica, informar a las personas cuando corresponda, limitar la finalidad y la conservación, evaluar los riesgos y consultar a su responsable de privacidad/asesoría. El proyecto no sustituye una evaluación de impacto cuando resulte necesaria.

## Limitaciones conocidas

- La protección de fórmula se aplica a exportaciones de esta aplicación; no modifica el CSV de origen.
- Los límites de tamaño/filas/columnas reducen algunos riesgos de consumo de recursos, pero no hacen que un archivo desconocido sea inherentemente seguro.
- La app no tiene autenticación, roles, auditoría ni cifrado propio de archivos en reposo.
- La importancia de variables no demuestra causalidad y las métricas del informe original no están garantizadas en otros conjuntos de datos.
