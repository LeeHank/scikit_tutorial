--- 
title: "A Minimal Book Example"
author: "John Doe"
date: "2022-01-17"
site: bookdown::bookdown_site
documentclass: book
bibliography: [book.bib, packages.bib]
# url: your book url like https://bookdown.org/yihui/bookdown
# cover-image: path to the social sharing image like images/cover.jpg
description: |
  This is a minimal example of using the bookdown package to write a book.
  The HTML output format for this example is bookdown::bs4_book,
  set in the _output.yml file.
biblio-style: apalike
csl: chicago-fullnote-bibliography.csl
---

# About {-}

# 環境準備 {-}  

* 當初是用 renv 專案，再加上 `renv::use_python()` 來開啟 python 的 virtual environment。那開完後，他的虛擬環境是建立在這個路徑： `renv/python/virtualenvs/renv-python-3.8.0`。所以虛擬資料夾名稱叫 `renv-python-3.8.0`。這就像自己用 VSCode 開虛擬環境時，會寫 `python -m venv my_env`，那時的 my_env 就像現在的 renv-python-3.8.0。  
* 如果用 RStudio 開啟此專案，那 python 的 virtual environment 會自動被 activate，所以就正常使用 python 是沒問題  
* 但如果用 VSCode 開啟此專案，就需要手動做一些事：  
  * 打開 terminal，先 activate 虛擬環境： `source renv/python/virtualenvs/renv-python-3.8.0/bin/activate`  
  * 第一次做的時候，為了讓 jupyter notebook 也可以吃到虛擬環境，所以要加做：  
    * `pip install ipykernel`  
    * `ipython kernel install --name=renv-python-3.8.0`
  * 之後就正常打開 jupyter notebook，然後 `ctrl+shift+p`，打 `python: select interpreter`，然後選到虛擬環境的 python 位置： `此專案根目錄/renv/virtualenvs/renv-python-3.8.0/bin/python`。這樣，這個 jupyter notebook 就會用虛擬環境的 python 和 package 來執行  
  * 如果要確認現在的 jupyter notebook 有沒有吃到虛擬環境，可以打 `! which python` (驚嘆號是讓 code chunk 可以執行 bash 指令)，那路徑應該要是剛剛寫的： `此專案根目錄/renv/virtualenvs/renv-python-3.8.0/bin/python`
