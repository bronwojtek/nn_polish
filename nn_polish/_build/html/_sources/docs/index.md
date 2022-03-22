

<!-- #region -->
# Sieci neuronowe dla początkujących w Pythonie: wykłady w Jupyter Book


[**Wojciech Broniowski**](https://www.ujk.edu.pl/~broniows)

<!--
[**Jan Kochanowski University**](https://www.ujk.edu.pl), Kielce, Poland, and

[**Institute of Nuclear Physics PAN**](https://www.ifj.edu.pl), Kraków
-->

<!-- #endregion -->

Niniejsze wykłady były pierwotnie prowadzone dla studentów inżynierii danych na  [Uniwersytecie Jana Kochanowskiego](https://www.ujk.edu.pl) w Kielcach i dla [Krakowskiej Szkoły Interdyscyplinarnych Studiów Doktoranckich](https://kisd.ifj.edu.pl/news/). Wyjaśniają bardzo podstawowe koncepcje sieci neuronowych na najbardziej przystępnym poziomie, wymagając od studenta jedynie bardzo podstawowej znajomości Pythona, a właściwie dowolnego języka programowania. W trosce o prostotę, kod dla różnych algorytmów sieci neuronowych pisany jest od podstaw, tj. bez użycia dedykowanych bibliotek wyższego poziomu. W ten sposób można dokładnie prześledzić wszystkie etapy programowania.

```{admonition} Zwięzłość
:class: note

Tekst jest zwięzły (wydruk pdf ma ~130 stron wraz z załącznikami), więc pilny student może ukończyć kurs w kilka popołudni!
```


```{admonition} Linki
:class: tip

- Jupyter Book: 
[https://bronwojtek.github.io/nn_polish/docs/index.html](https://bronwojtek.github.io/nn_polish/docs/index.html)

<!---
- pdf i kody: [www.ifj.edu.pl/~broniows/nn_polish](https://www.ifj.edu.pl/~broniows/nn_polish) lub [www.ujk.edu.pl/~broniows/nn_polish](https://www.ujk.edu.pl/~broniows/nn_polish)
--->

Pierwotna angielska wersja książki:

- Jupyter Book: 
[https://bronwojtek.github.io/neuralnets-in-raw-python/docs/index.html](https://bronwojtek.github.io/neuralnets-in-raw-python/docs/index.html)

- pdf i kody: [www.ifj.edu.pl/~broniows/nn](https://www.ifj.edu.pl/~broniows/nn) lub [www.ujk.edu.pl/~broniows/nn](https://www.ujk.edu.pl/~broniows/nn)

```


```{admonition} Jak uruchamiać kody w książce
:class: important

Główną zaletą książek wykonywalnych jest to, że czytelnik może cieszyć się z samodzielnego uruchamiania kodów źródłowych, modyfikowania ich, czy zabawy z parametrami. Nie jest potrzebne pobieranie, instalacja ani konfiguracja. Po prostu przejdź do

[https://bronwojtek.github.io/nn_polish/docs/index.html](https://bronwojtek.github.io/nn_polish/docs/index.html),

w menu po lewej stronie wybierz dowolny rozdział poniżej Wstępu, kliknij ikonę „rakiety” w prawym górnym rogu ekranu i wybierz „Colab” lub „Binder”. Po pewnym czasie inicjalizacji (za pierwszym razem dla Bindera trwa to dość długo) można uruchomić notebook.

Dla wykonywania lokalnego, kody dla każdego rozdziału w postaci
notebooków [Jupytera](https://jupyter.org) można pobrać klikając ikonę „strzałki w dół” w prawym górnym rogu ekranu. Pełen zestaw plików jest również dostępny z linków podanych powyżej.

Dodatek {ref}`app-run` wyjaśnia, jak postępować przy lokalnym wykonywaniu programów.
```

```{admonition} $~$
Książka wykonywalna, utworzona przez oprogramowanie [Jupyter Book
2.0](https://beta.jupyterbook.org/intro.html), będące częścią
[ExecutableBookProject](https://ebp.jupyterbook.org/en/latest/).  
```

<!--

ISBN: **978-83-962099-0-0** (wersja pdf) 

![](images/barcode.png)

-->
