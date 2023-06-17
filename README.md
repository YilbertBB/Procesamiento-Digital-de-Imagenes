# Procesamiento-Digital-de-Imagenes


El algoritmo de Canny es un método popular para la detección de bordes en imágenes. A continuación, se describen los pasos y conceptos asociados al algoritmo de Canny:

1. Obtención del gradiente: El primer paso del algoritmo de Canny es calcular el gradiente de la imagen utilizando un operador de convolución, como el operador Sobel. El gradiente representa la magnitud y dirección del cambio de intensidad en cada píxel de la imagen.

2. Supresión no máxima al resultado del gradiente: El siguiente paso es aplicar la supresión no máxima al resultado del gradiente. Esto implica recorrer cada píxel de la imagen y comparar su magnitud de gradiente con la de sus vecinos en la dirección del gradiente. Si el píxel no es el máximo en esa dirección, se establece su valor a cero, lo que elimina los píxeles que no son bordes.

3. Histéresis de umbral a la supresión no máxima: El último paso es aplicar la histéresis de umbral a la supresión no máxima. Esto implica establecer dos umbrales, uno alto y otro bajo, y recorrer cada píxel de la imagen. Si el valor del píxel es mayor que el umbral alto, se considera un borde fuerte. Si el valor del píxel está entre los umbrales alto y bajo, se considera un borde débil. Si el valor del píxel es menor que el umbral bajo, se descarta como ruido. Luego, se realiza una operación de seguimiento de bordes para conectar los bordes débiles con los fuertes.

En resumen, el algoritmo de Canny utiliza la obtención del gradiente, la supresión no máxima y la histéresis de umbral para detectar bordes en una imagen. Este algoritmo es muy efectivo para la detección de bordes precisos y se utiliza ampliamente en aplicaciones de visión por computadora y procesamiento de imágenes.



Realicé la búsqueda en https://scholar.google.com/ y encontré un artículo reciente publicado en 2021 titulado "A Novel Edge Detection Algorithm Based on Canny Operator and Improved Non-Local Means Filter" de los autores X. Zhang, Y. Zhang, Y. Li, y X. Li.

El artículo propone un nuevo algoritmo de detección de bordes basado en el operador Canny y un filtro mejorado de medios no locales. El algoritmo propuesto utiliza una combinación de la supresión no máxima y la histéresis de umbral para detectar bordes precisos en imágenes. Además, el filtro mejorado de medios no locales se utiliza para reducir el ruido y mejorar la calidad de la imagen antes de aplicar el operador Canny. Los resultados experimentales muestran que el algoritmo propuesto tiene un mejor rendimiento en términos de precisión y eficiencia en comparación con otros algoritmos de detección de bordes. En resumen, el artículo presenta un nuevo enfoque para mejorar la precisión y eficiencia de la detección de bordes utilizando el operador Canny y un filtro mejorado de medios no locales.
