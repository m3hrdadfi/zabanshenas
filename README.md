<h1 align="center">Zabanshenas 🕵</h1>
<p align="center"><a href="https://doi.org/10.5281/zenodo.5029022"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.5029022.svg" alt="DOI"></a></p>
<br/><br/>

A Transformer-based solution for identifying the most likely language of a written document/text. 
**Zabanshenas** is a Persian word that has two meanings:

- A person who studies linguistics.
- A way to identify the type of written language.

<br/>


## Introduction 

In this repository, I will use another perspective in creating a language detection model using [Transformers](https://arxiv.org/abs/1706.03762). Nowadays, Transformers have played a massive role in Natural Language Processing fields. In short, Transformers uses an attention mechanism to boost the speed and extract a high level of information (abstraction).

There are plenty of ways, solutions, and packages to find the language of a written piece of text or document. All of them have their pros and cons. Some able to detect faster and support as many languages as possible. However, in this case, I intend to use Transformers to understand similar groups of languages and cover 235 languages thanks to [WiLI-2018](https://arxiv.org/abs/1801.07779) and the Transformer architecture.

This model can detect a written language in three different stages: paragraph, sentence, and subset of text between three and four tokens.

## WilI-2018 (Cleaned version)

It is a benchmark for monolingual written natural language identification (high covering of a specific language). It contains 229,095 paragraphs that cover 235 languages. Language distribution includes 122 Indo-European languages, 22 Austronesian languages, 17 Turkic languages, 14 Uralic languages, 11 Niger-Congo languages, 10 Sino-Tibetan languages, 9 Afro-Asiatic languages, 6 constructed languages, and 24 languages of smaller families. It also consists of groups of similar languages and dialects:
- Arabic, Egyptian Arabic
- English, Old English, Scots
- Standard Chinese, Min Nan Chinese, Hakka Chinese, Literary Chinese, Wu Chinese
- German, Bavarian, Low German, Palatine German, Ripuarisch, Alemannic German, Pennsylvania German
- Belarusian, Belarusian (Taraschkewiza)
- Kurdish, Central Kurdish
- Indonesian, Minangkabau, Banyumasan, Banjar, Sundanese, Javanese
- Languages are spoken in India:
    - Maithili, Bhojpuri
    - Bengali, Bishnupriya
    - Konkani, Marathi
- Russian, Komi-Permyak
- Persian, Gilaki, Mazanderani

On the other hand, this dataset addresses low-resources languages, as shown in Fig 1: 
- Chechen
- Haitian Creole
- Newari
- Pampanga 

<details>
<summary>More info:</summary>

<p align="center">
    <img src="reports/samples_per_lang.png" alt="">
    <br>
    <em>Fig 1: The distribution of samples per language.</em>
</p>

As well as minor textual snippet languages (word level, character level), as shown in Fig 2, 3:

- Word Level
    - Literary Chinese
    - Japanese
    - Standard Chinese
    - Wu Chinese
    - Chechen
- Character Level:
    - Chechen
    - Haitian Creole
    - Newari
    - Minangkabau
    - Pampanga

<p align="center">
    <img src="reports/words_per_lang.png" alt="">
    <br>
    <em>Fig 2: The distribution of words per language.</em>
</p>

<p align="center">
    <img src="reports/chars_per_lang.png" alt="">
    <br>
    <em>Fig 3: The distribution of characters per language.</em>
</p>


Instead of using a word or character level of tokenization, I used subword tokenization [BPE](https://arxiv.org/abs/1909.03341) with max sequence length of 512. It allows the Transformer to have a rich vocabulary size while learning meaningful context-independent representations. The distribution of vocabulary in WiLI-2018 is shown in Fig 4.

<p align="center">
    <img src="reports/tokens_per_lang.png" alt="">
    <br>
    <em>Fig 4: The distribution of BPE-tokens per language.</em>
</p>

</details>

## Model (Architecture)
[RoBERTa](https://arxiv.org/abs/1907.11692) as a candidate model is used for this specific task with the following procedure. 
Firstly, the model is trained on the WILI-2018 corpus and then fine-tuned as a SequenceClassification task to detect independent and similar languages and dialects. 
The data is divided into three segments, 70% of the whole data (train + test) as paragraph choices, 15% tokenized into sentences, and what remains, split into the subset of three or five tokens per record to achieve better results.

**In total, the dataset consists of a 1M training set, 11K validation set, and 11K testing set.**

## Evaluation

Precision, recall, F1 scores for each language and level are presented in the following tables.

<details>
<summary>Paragraph level</summary>

|                language                | precision |  recall  | f1-score |
|:--------------------------------------:|:---------:|:--------:|:--------:|
|             Achinese (ace)             |  1.000000 | 0.982143 | 0.990991 |
|             Afrikaans (afr)            |  1.000000 | 1.000000 | 1.000000 |
|         Alemannic German (als)         |  1.000000 | 0.946429 | 0.972477 |
|              Amharic (amh)             |  1.000000 | 0.982143 | 0.990991 |
|            Old English (ang)           |  0.981818 | 0.964286 | 0.972973 |
|              Arabic (ara)              |  0.846154 | 0.982143 | 0.909091 |
|             Aragonese (arg)            |  1.000000 | 1.000000 | 1.000000 |
|          Egyptian Arabic (arz)         |  0.979592 | 0.857143 | 0.914286 |
|             Assamese (asm)             |  0.981818 | 0.964286 | 0.972973 |
|             Asturian (ast)             |  0.964912 | 0.982143 | 0.973451 |
|               Avar (ava)               |  0.941176 | 0.905660 | 0.923077 |
|              Aymara (aym)              |  0.964912 | 0.982143 | 0.973451 |
|         South Azerbaijani (azb)        |  0.965517 | 1.000000 | 0.982456 |
|            Azerbaijani (aze)           |  1.000000 | 1.000000 | 1.000000 |
|              Bashkir (bak)             |  1.000000 | 0.978261 | 0.989011 |
|             Bavarian (bar)             |  0.843750 | 0.964286 | 0.900000 |
|           Central Bikol (bcl)          |  1.000000 | 0.982143 | 0.990991 |
| Belarusian (Taraschkewiza) (be-tarask) |  1.000000 | 0.875000 | 0.933333 |
|            Belarusian (bel)            |  0.870968 | 0.964286 | 0.915254 |
|              Bengali (ben)             |  0.982143 | 0.982143 | 0.982143 |
|             Bhojpuri (bho)             |  1.000000 | 0.928571 | 0.962963 |
|              Banjar (bjn)              |  0.981132 | 0.945455 | 0.962963 |
|              Tibetan (bod)             |  1.000000 | 0.982143 | 0.990991 |
|              Bosnian (bos)             |  0.552632 | 0.375000 | 0.446809 |
|            Bishnupriya (bpy)           |  1.000000 | 0.982143 | 0.990991 |
|              Breton (bre)              |  1.000000 | 0.964286 | 0.981818 |
|             Bulgarian (bul)            |  1.000000 | 0.964286 | 0.981818 |
|              Buryat (bxr)              |  0.946429 | 0.946429 | 0.946429 |
|              Catalan (cat)             |  0.982143 | 0.982143 | 0.982143 |
|             Chavacano (cbk)            |  0.914894 | 0.767857 | 0.834951 |
|             Min Dong (cdo)             |  1.000000 | 0.982143 | 0.990991 |
|              Cebuano (ceb)             |  1.000000 | 1.000000 | 1.000000 |
|               Czech (ces)              |  1.000000 | 1.000000 | 1.000000 |
|              Chechen (che)             |  1.000000 | 1.000000 | 1.000000 |
|             Cherokee (chr)             |  1.000000 | 0.963636 | 0.981481 |
|              Chuvash (chv)             |  0.938776 | 0.958333 | 0.948454 |
|          Central Kurdish (ckb)         |  1.000000 | 1.000000 | 1.000000 |
|              Cornish (cor)             |  1.000000 | 1.000000 | 1.000000 |
|             Corsican (cos)             |  1.000000 | 0.982143 | 0.990991 |
|           Crimean Tatar (crh)          |  1.000000 | 0.946429 | 0.972477 |
|             Kashubian (csb)            |  1.000000 | 0.963636 | 0.981481 |
|               Welsh (cym)              |  1.000000 | 1.000000 | 1.000000 |
|              Danish (dan)              |  1.000000 | 1.000000 | 1.000000 |
|              German (deu)              |  0.828125 | 0.946429 | 0.883333 |
|               Dimli (diq)              |  0.964912 | 0.982143 | 0.973451 |
|              Dhivehi (div)             |  1.000000 | 1.000000 | 1.000000 |
|           Lower Sorbian (dsb)          |  1.000000 | 0.982143 | 0.990991 |
|              Doteli (dty)              |  0.940000 | 0.854545 | 0.895238 |
|              Emilian (egl)             |  1.000000 | 0.928571 | 0.962963 |
|           Modern Greek (ell)           |  1.000000 | 1.000000 | 1.000000 |
|              English (eng)             |  0.588889 | 0.946429 | 0.726027 |
|             Esperanto (epo)            |  1.000000 | 0.982143 | 0.990991 |
|             Estonian (est)             |  0.963636 | 0.946429 | 0.954955 |
|              Basque (eus)              |  1.000000 | 0.982143 | 0.990991 |
|           Extremaduran (ext)           |  0.982143 | 0.982143 | 0.982143 |
|              Faroese (fao)             |  1.000000 | 1.000000 | 1.000000 |
|              Persian (fas)             |  0.948276 | 0.982143 | 0.964912 |
|              Finnish (fin)             |  1.000000 | 1.000000 | 1.000000 |
|              French (fra)              |  0.710145 | 0.875000 | 0.784000 |
|              Arpitan (frp)             |  1.000000 | 0.946429 | 0.972477 |
|          Western Frisian (fry)         |  0.982143 | 0.982143 | 0.982143 |
|             Friulian (fur)             |  1.000000 | 0.982143 | 0.990991 |
|              Gagauz (gag)              |  0.981132 | 0.945455 | 0.962963 |
|          Scottish Gaelic (gla)         |  0.982143 | 0.982143 | 0.982143 |
|               Irish (gle)              |  0.949153 | 1.000000 | 0.973913 |
|             Galician (glg)             |  1.000000 | 1.000000 | 1.000000 |
|              Gilaki (glk)              |  0.981132 | 0.945455 | 0.962963 |
|               Manx (glv)               |  1.000000 | 1.000000 | 1.000000 |
|              Guarani (grn)             |  1.000000 | 0.964286 | 0.981818 |
|             Gujarati (guj)             |  1.000000 | 0.982143 | 0.990991 |
|           Hakka Chinese (hak)          |  0.981818 | 0.964286 | 0.972973 |
|          Haitian Creole (hat)          |  1.000000 | 1.000000 | 1.000000 |
|               Hausa (hau)              |  1.000000 | 0.945455 | 0.971963 |
|          Serbo-Croatian (hbs)          |  0.448276 | 0.464286 | 0.456140 |
|              Hebrew (heb)              |  1.000000 | 0.982143 | 0.990991 |
|            Fiji Hindi (hif)            |  0.890909 | 0.890909 | 0.890909 |
|               Hindi (hin)              |  0.981481 | 0.946429 | 0.963636 |
|             Croatian (hrv)             |  0.500000 | 0.636364 | 0.560000 |
|           Upper Sorbian (hsb)          |  0.955556 | 1.000000 | 0.977273 |
|             Hungarian (hun)            |  1.000000 | 1.000000 | 1.000000 |
|             Armenian (hye)             |  1.000000 | 0.981818 | 0.990826 |
|               Igbo (ibo)               |  0.918033 | 1.000000 | 0.957265 |
|                Ido (ido)               |  1.000000 | 1.000000 | 1.000000 |
|            Interlingue (ile)           |  1.000000 | 0.962264 | 0.980769 |
|               Iloko (ilo)              |  0.947368 | 0.964286 | 0.955752 |
|            Interlingua (ina)           |  1.000000 | 1.000000 | 1.000000 |
|            Indonesian (ind)            |  0.761905 | 0.872727 | 0.813559 |
|             Icelandic (isl)            |  1.000000 | 1.000000 | 1.000000 |
|              Italian (ita)             |  0.861538 | 1.000000 | 0.925620 |
|          Jamaican Patois (jam)         |  1.000000 | 0.946429 | 0.972477 |
|             Javanese (jav)             |  0.964912 | 0.982143 | 0.973451 |
|              Lojban (jbo)              |  1.000000 | 1.000000 | 1.000000 |
|             Japanese (jpn)             |  1.000000 | 1.000000 | 1.000000 |
|            Karakalpak (kaa)            |  0.965517 | 1.000000 | 0.982456 |
|              Kabyle (kab)              |  1.000000 | 0.964286 | 0.981818 |
|              Kannada (kan)             |  0.982143 | 0.982143 | 0.982143 |
|             Georgian (kat)             |  1.000000 | 0.964286 | 0.981818 |
|              Kazakh (kaz)              |  0.980769 | 0.980769 | 0.980769 |
|             Kabardian (kbd)            |  1.000000 | 0.982143 | 0.990991 |
|           Central Khmer (khm)          |  0.960784 | 0.875000 | 0.915888 |
|            Kinyarwanda (kin)           |  0.981132 | 0.928571 | 0.954128 |
|              Kirghiz (kir)             |  1.000000 | 1.000000 | 1.000000 |
|           Komi-Permyak (koi)           |  0.962264 | 0.910714 | 0.935780 |
|              Konkani (kok)             |  0.964286 | 0.981818 | 0.972973 |
|               Komi (kom)               |  1.000000 | 0.962264 | 0.980769 |
|              Korean (kor)              |  1.000000 | 1.000000 | 1.000000 |
|          Karachay-Balkar (krc)         |  1.000000 | 0.982143 | 0.990991 |
|            Ripuarisch (ksh)            |  1.000000 | 0.964286 | 0.981818 |
|              Kurdish (kur)             |  1.000000 | 0.964286 | 0.981818 |
|              Ladino (lad)              |  1.000000 | 1.000000 | 1.000000 |
|                Lao (lao)               |  0.961538 | 0.909091 | 0.934579 |
|               Latin (lat)              |  0.877193 | 0.943396 | 0.909091 |
|              Latvian (lav)             |  0.963636 | 0.946429 | 0.954955 |
|             Lezghian (lez)             |  1.000000 | 0.964286 | 0.981818 |
|             Ligurian (lij)             |  1.000000 | 0.964286 | 0.981818 |
|             Limburgan (lim)            |  0.938776 | 1.000000 | 0.968421 |
|              Lingala (lin)             |  0.980769 | 0.927273 | 0.953271 |
|            Lithuanian (lit)            |  0.982456 | 1.000000 | 0.991150 |
|              Lombard (lmo)             |  1.000000 | 1.000000 | 1.000000 |
|           Northern Luri (lrc)          |  1.000000 | 0.928571 | 0.962963 |
|             Latgalian (ltg)            |  1.000000 | 0.982143 | 0.990991 |
|           Luxembourgish (ltz)          |  0.949153 | 1.000000 | 0.973913 |
|              Luganda (lug)             |  1.000000 | 1.000000 | 1.000000 |
|         Literary Chinese (lzh)         |  1.000000 | 1.000000 | 1.000000 |
|             Maithili (mai)             |  0.931034 | 0.964286 | 0.947368 |
|             Malayalam (mal)            |  1.000000 | 0.982143 | 0.990991 |
|          Banyumasan (map-bms)          |  0.977778 | 0.785714 | 0.871287 |
|              Marathi (mar)             |  0.949153 | 1.000000 | 0.973913 |
|              Moksha (mdf)              |  0.980000 | 0.890909 | 0.933333 |
|           Eastern Mari (mhr)           |  0.981818 | 0.964286 | 0.972973 |
|            Minangkabau (min)           |  1.000000 | 1.000000 | 1.000000 |
|            Macedonian (mkd)            |  1.000000 | 0.981818 | 0.990826 |
|             Malagasy (mlg)             |  0.981132 | 1.000000 | 0.990476 |
|              Maltese (mlt)             |  0.982456 | 1.000000 | 0.991150 |
|          Min Nan Chinese (nan)         |  1.000000 | 1.000000 | 1.000000 |
|             Mongolian (mon)            |  1.000000 | 0.981818 | 0.990826 |
|               Maori (mri)              |  1.000000 | 1.000000 | 1.000000 |
|           Western Mari (mrj)           |  0.982456 | 1.000000 | 0.991150 |
|               Malay (msa)              |  0.862069 | 0.892857 | 0.877193 |
|             Mirandese (mwl)            |  1.000000 | 0.982143 | 0.990991 |
|              Burmese (mya)             |  1.000000 | 1.000000 | 1.000000 |
|               Erzya (myv)              |  0.818182 | 0.964286 | 0.885246 |
|            Mazanderani (mzn)           |  0.981481 | 1.000000 | 0.990654 |
|            Neapolitan (nap)            |  1.000000 | 0.981818 | 0.990826 |
|              Navajo (nav)              |  1.000000 | 1.000000 | 1.000000 |
|         Classical Nahuatl (nci)        |  0.981481 | 0.946429 | 0.963636 |
|            Low German (nds)            |  0.982143 | 0.982143 | 0.982143 |
|        West Low German (nds-nl)        |  1.000000 | 1.000000 | 1.000000 |
|      Nepali (macrolanguage) (nep)      |  0.881356 | 0.928571 | 0.904348 |
|              Newari (new)              |  1.000000 | 0.909091 | 0.952381 |
|               Dutch (nld)              |  0.982143 | 0.982143 | 0.982143 |
|         Norwegian Nynorsk (nno)        |  1.000000 | 1.000000 | 1.000000 |
|              Bokmål (nob)              |  1.000000 | 1.000000 | 1.000000 |
|               Narom (nrm)              |  0.981818 | 0.964286 | 0.972973 |
|          Northern Sotho (nso)          |  1.000000 | 1.000000 | 1.000000 |
|              Occitan (oci)             |  0.903846 | 0.839286 | 0.870370 |
|          Livvi-Karelian (olo)          |  0.982456 | 1.000000 | 0.991150 |
|               Oriya (ori)              |  0.964912 | 0.982143 | 0.973451 |
|               Oromo (orm)              |  0.982143 | 0.982143 | 0.982143 |
|             Ossetian (oss)             |  0.982143 | 1.000000 | 0.990991 |
|            Pangasinan (pag)            |  0.980000 | 0.875000 | 0.924528 |
|             Pampanga (pam)             |  0.928571 | 0.896552 | 0.912281 |
|              Panjabi (pan)             |  1.000000 | 1.000000 | 1.000000 |
|            Papiamento (pap)            |  1.000000 | 0.964286 | 0.981818 |
|              Picard (pcd)              |  0.849057 | 0.849057 | 0.849057 |
|        Pennsylvania German (pdc)       |  0.854839 | 0.946429 | 0.898305 |
|          Palatine German (pfl)         |  0.946429 | 0.946429 | 0.946429 |
|          Western Panjabi (pnb)         |  0.981132 | 0.962963 | 0.971963 |
|              Polish (pol)              |  0.933333 | 1.000000 | 0.965517 |
|            Portuguese (por)            |  0.774648 | 0.982143 | 0.866142 |
|              Pushto (pus)              |  1.000000 | 0.910714 | 0.953271 |
|              Quechua (que)             |  0.962963 | 0.928571 | 0.945455 |
|      Tarantino dialect (roa-tara)      |  1.000000 | 0.964286 | 0.981818 |
|              Romansh (roh)             |  1.000000 | 0.928571 | 0.962963 |
|             Romanian (ron)             |  0.965517 | 1.000000 | 0.982456 |
|               Rusyn (rue)              |  0.946429 | 0.946429 | 0.946429 |
|             Aromanian (rup)            |  0.962963 | 0.928571 | 0.945455 |
|              Russian (rus)             |  0.859375 | 0.982143 | 0.916667 |
|               Yakut (sah)              |  1.000000 | 0.982143 | 0.990991 |
|             Sanskrit (san)             |  0.982143 | 0.982143 | 0.982143 |
|             Sicilian (scn)             |  1.000000 | 1.000000 | 1.000000 |
|               Scots (sco)              |  0.982143 | 0.982143 | 0.982143 |
|            Samogitian (sgs)            |  1.000000 | 0.982143 | 0.990991 |
|              Sinhala (sin)             |  0.964912 | 0.982143 | 0.973451 |
|              Slovak (slk)              |  1.000000 | 0.982143 | 0.990991 |
|              Slovene (slv)             |  1.000000 | 0.981818 | 0.990826 |
|           Northern Sami (sme)          |  0.962264 | 0.962264 | 0.962264 |
|               Shona (sna)              |  0.933333 | 1.000000 | 0.965517 |
|              Sindhi (snd)              |  1.000000 | 1.000000 | 1.000000 |
|              Somali (som)              |  0.948276 | 1.000000 | 0.973451 |
|              Spanish (spa)             |  0.739130 | 0.910714 | 0.816000 |
|             Albanian (sqi)             |  0.982143 | 0.982143 | 0.982143 |
|             Sardinian (srd)            |  1.000000 | 0.982143 | 0.990991 |
|              Sranan (srn)              |  1.000000 | 1.000000 | 1.000000 |
|              Serbian (srp)             |  1.000000 | 0.946429 | 0.972477 |
|          Saterfriesisch (stq)          |  1.000000 | 0.964286 | 0.981818 |
|             Sundanese (sun)            |  1.000000 | 0.977273 | 0.988506 |
|      Swahili (macrolanguage) (swa)     |  1.000000 | 1.000000 | 1.000000 |
|              Swedish (swe)             |  1.000000 | 1.000000 | 1.000000 |
|             Silesian (szl)             |  1.000000 | 0.981481 | 0.990654 |
|               Tamil (tam)              |  0.982143 | 1.000000 | 0.990991 |
|               Tatar (tat)              |  1.000000 | 1.000000 | 1.000000 |
|               Tulu (tcy)               |  0.982456 | 1.000000 | 0.991150 |
|              Telugu (tel)              |  1.000000 | 0.920000 | 0.958333 |
|               Tetum (tet)              |  1.000000 | 0.964286 | 0.981818 |
|               Tajik (tgk)              |  1.000000 | 1.000000 | 1.000000 |
|              Tagalog (tgl)             |  1.000000 | 1.000000 | 1.000000 |
|               Thai (tha)               |  0.932203 | 0.982143 | 0.956522 |
|              Tongan (ton)              |  1.000000 | 0.964286 | 0.981818 |
|              Tswana (tsn)              |  1.000000 | 1.000000 | 1.000000 |
|              Turkmen (tuk)             |  1.000000 | 0.982143 | 0.990991 |
|              Turkish (tur)             |  0.901639 | 0.982143 | 0.940171 |
|               Tuvan (tyv)              |  1.000000 | 0.964286 | 0.981818 |
|              Udmurt (udm)              |  1.000000 | 0.982143 | 0.990991 |
|              Uighur (uig)              |  1.000000 | 0.982143 | 0.990991 |
|             Ukrainian (ukr)            |  0.963636 | 0.946429 | 0.954955 |
|               Urdu (urd)               |  1.000000 | 0.982143 | 0.990991 |
|               Uzbek (uzb)              |  1.000000 | 1.000000 | 1.000000 |
|             Venetian (vec)             |  1.000000 | 0.982143 | 0.990991 |
|               Veps (vep)               |  0.982456 | 1.000000 | 0.991150 |
|            Vietnamese (vie)            |  0.964912 | 0.982143 | 0.973451 |
|              Vlaams (vls)              |  1.000000 | 0.982143 | 0.990991 |
|              Volapük (vol)             |  1.000000 | 1.000000 | 1.000000 |
|               Võro (vro)               |  0.964286 | 0.964286 | 0.964286 |
|               Waray (war)              |  1.000000 | 0.982143 | 0.990991 |
|              Walloon (wln)             |  1.000000 | 1.000000 | 1.000000 |
|               Wolof (wol)              |  0.981481 | 0.963636 | 0.972477 |
|            Wu Chinese (wuu)            |  0.981481 | 0.946429 | 0.963636 |
|               Xhosa (xho)              |  1.000000 | 0.964286 | 0.981818 |
|            Mingrelian (xmf)            |  1.000000 | 0.964286 | 0.981818 |
|              Yiddish (yid)             |  1.000000 | 1.000000 | 1.000000 |
|              Yoruba (yor)              |  0.964912 | 0.982143 | 0.973451 |
|              Zeeuws (zea)              |  1.000000 | 0.982143 | 0.990991 |
|           Cantonese (zh-yue)           |  0.981481 | 0.946429 | 0.963636 |
|         Standard Chinese (zho)         |  0.932203 | 0.982143 | 0.956522 |
|                accuracy                |  0.963055 | 0.963055 | 0.963055 |
|                macro avg               |  0.966424 | 0.963216 | 0.963891 |
|              weighted avg              |  0.966040 | 0.963055 | 0.963606 |

</details>

<details>
<summary>Sentence level</summary>

|                language                | precision |  recall  | f1-score |
|:--------------------------------------:|:---------:|:--------:|:--------:|
|             Achinese (ace)             |  0.754545 | 0.873684 | 0.809756 |
|             Afrikaans (afr)            |  0.708955 | 0.940594 | 0.808511 |
|         Alemannic German (als)         |  0.870130 | 0.752809 | 0.807229 |
|              Amharic (amh)             |  1.000000 | 0.820000 | 0.901099 |
|            Old English (ang)           |  0.966667 | 0.906250 | 0.935484 |
|              Arabic (ara)              |  0.907692 | 0.967213 | 0.936508 |
|             Aragonese (arg)            |  0.921569 | 0.959184 | 0.940000 |
|          Egyptian Arabic (arz)         |  0.964286 | 0.843750 | 0.900000 |
|             Assamese (asm)             |  0.964286 | 0.870968 | 0.915254 |
|             Asturian (ast)             |  0.880000 | 0.795181 | 0.835443 |
|               Avar (ava)               |  0.864198 | 0.843373 | 0.853659 |
|              Aymara (aym)              |  1.000000 | 0.901961 | 0.948454 |
|         South Azerbaijani (azb)        |  0.979381 | 0.989583 | 0.984456 |
|            Azerbaijani (aze)           |  0.989899 | 0.960784 | 0.975124 |
|              Bashkir (bak)             |  0.837209 | 0.857143 | 0.847059 |
|             Bavarian (bar)             |  0.741935 | 0.766667 | 0.754098 |
|           Central Bikol (bcl)          |  0.962963 | 0.928571 | 0.945455 |
| Belarusian (Taraschkewiza) (be-tarask) |  0.857143 | 0.733333 | 0.790419 |
|            Belarusian (bel)            |  0.775510 | 0.752475 | 0.763819 |
|              Bengali (ben)             |  0.861111 | 0.911765 | 0.885714 |
|             Bhojpuri (bho)             |  0.965517 | 0.933333 | 0.949153 |
|              Banjar (bjn)              |  0.891566 | 0.880952 | 0.886228 |
|              Tibetan (bod)             |  1.000000 | 1.000000 | 1.000000 |
|              Bosnian (bos)             |  0.375000 | 0.323077 | 0.347107 |
|            Bishnupriya (bpy)           |  0.986301 | 1.000000 | 0.993103 |
|              Breton (bre)              |  0.951613 | 0.893939 | 0.921875 |
|             Bulgarian (bul)            |  0.945055 | 0.877551 | 0.910053 |
|              Buryat (bxr)              |  0.955556 | 0.843137 | 0.895833 |
|              Catalan (cat)             |  0.692308 | 0.750000 | 0.720000 |
|             Chavacano (cbk)            |  0.842857 | 0.641304 | 0.728395 |
|             Min Dong (cdo)             |  0.972973 | 1.000000 | 0.986301 |
|              Cebuano (ceb)             |  0.981308 | 0.954545 | 0.967742 |
|               Czech (ces)              |  0.944444 | 0.915385 | 0.929687 |
|              Chechen (che)             |  0.875000 | 0.700000 | 0.777778 |
|             Cherokee (chr)             |  1.000000 | 0.970588 | 0.985075 |
|              Chuvash (chv)             |  0.875000 | 0.836957 | 0.855556 |
|          Central Kurdish (ckb)         |  1.000000 | 0.983051 | 0.991453 |
|              Cornish (cor)             |  0.979592 | 0.969697 | 0.974619 |
|             Corsican (cos)             |  0.986842 | 0.925926 | 0.955414 |
|           Crimean Tatar (crh)          |  0.958333 | 0.907895 | 0.932432 |
|             Kashubian (csb)            |  0.920354 | 0.904348 | 0.912281 |
|               Welsh (cym)              |  0.971014 | 0.943662 | 0.957143 |
|              Danish (dan)              |  0.865169 | 0.777778 | 0.819149 |
|              German (deu)              |  0.721311 | 0.822430 | 0.768559 |
|               Dimli (diq)              |  0.915966 | 0.923729 | 0.919831 |
|              Dhivehi (div)             |  1.000000 | 0.991228 | 0.995595 |
|           Lower Sorbian (dsb)          |  0.898876 | 0.879121 | 0.888889 |
|              Doteli (dty)              |  0.821429 | 0.638889 | 0.718750 |
|              Emilian (egl)             |  0.988095 | 0.922222 | 0.954023 |
|           Modern Greek (ell)           |  0.988636 | 0.966667 | 0.977528 |
|              English (eng)             |  0.522727 | 0.784091 | 0.627273 |
|             Esperanto (epo)            |  0.963855 | 0.930233 | 0.946746 |
|             Estonian (est)             |  0.922222 | 0.873684 | 0.897297 |
|              Basque (eus)              |  1.000000 | 0.941176 | 0.969697 |
|           Extremaduran (ext)           |  0.925373 | 0.885714 | 0.905109 |
|              Faroese (fao)             |  0.855072 | 0.887218 | 0.870849 |
|              Persian (fas)             |  0.879630 | 0.979381 | 0.926829 |
|              Finnish (fin)             |  0.952830 | 0.943925 | 0.948357 |
|              French (fra)              |  0.676768 | 0.943662 | 0.788235 |
|              Arpitan (frp)             |  0.867925 | 0.807018 | 0.836364 |
|          Western Frisian (fry)         |  0.956989 | 0.890000 | 0.922280 |
|             Friulian (fur)             |  1.000000 | 0.857143 | 0.923077 |
|              Gagauz (gag)              |  0.939024 | 0.802083 | 0.865169 |
|          Scottish Gaelic (gla)         |  1.000000 | 0.879121 | 0.935673 |
|               Irish (gle)              |  0.989247 | 0.958333 | 0.973545 |
|             Galician (glg)             |  0.910256 | 0.922078 | 0.916129 |
|              Gilaki (glk)              |  0.964706 | 0.872340 | 0.916201 |
|               Manx (glv)               |  1.000000 | 0.965517 | 0.982456 |
|              Guarani (grn)             |  0.983333 | 1.000000 | 0.991597 |
|             Gujarati (guj)             |  1.000000 | 0.991525 | 0.995745 |
|           Hakka Chinese (hak)          |  0.955224 | 0.955224 | 0.955224 |
|          Haitian Creole (hat)          |  0.833333 | 0.666667 | 0.740741 |
|               Hausa (hau)              |  0.936709 | 0.913580 | 0.925000 |
|          Serbo-Croatian (hbs)          |  0.452830 | 0.410256 | 0.430493 |
|              Hebrew (heb)              |  0.988235 | 0.976744 | 0.982456 |
|            Fiji Hindi (hif)            |  0.936709 | 0.840909 | 0.886228 |
|               Hindi (hin)              |  0.965517 | 0.756757 | 0.848485 |
|             Croatian (hrv)             |  0.443820 | 0.537415 | 0.486154 |
|           Upper Sorbian (hsb)          |  0.951613 | 0.830986 | 0.887218 |
|             Hungarian (hun)            |  0.854701 | 0.909091 | 0.881057 |
|             Armenian (hye)             |  1.000000 | 0.816327 | 0.898876 |
|               Igbo (ibo)               |  0.974359 | 0.926829 | 0.950000 |
|                Ido (ido)               |  0.975000 | 0.987342 | 0.981132 |
|            Interlingue (ile)           |  0.880597 | 0.921875 | 0.900763 |
|               Iloko (ilo)              |  0.882353 | 0.821918 | 0.851064 |
|            Interlingua (ina)           |  0.952381 | 0.895522 | 0.923077 |
|            Indonesian (ind)            |  0.606383 | 0.695122 | 0.647727 |
|             Icelandic (isl)            |  0.978261 | 0.882353 | 0.927835 |
|              Italian (ita)             |  0.910448 | 0.910448 | 0.910448 |
|          Jamaican Patois (jam)         |  0.988764 | 0.967033 | 0.977778 |
|             Javanese (jav)             |  0.903614 | 0.862069 | 0.882353 |
|              Lojban (jbo)              |  0.943878 | 0.929648 | 0.936709 |
|             Japanese (jpn)             |  1.000000 | 0.764706 | 0.866667 |
|            Karakalpak (kaa)            |  0.940171 | 0.901639 | 0.920502 |
|              Kabyle (kab)              |  0.985294 | 0.837500 | 0.905405 |
|              Kannada (kan)             |  0.975806 | 0.975806 | 0.975806 |
|             Georgian (kat)             |  0.953704 | 0.903509 | 0.927928 |
|              Kazakh (kaz)              |  0.934579 | 0.877193 | 0.904977 |
|             Kabardian (kbd)            |  0.987952 | 0.953488 | 0.970414 |
|           Central Khmer (khm)          |  0.928571 | 0.829787 | 0.876404 |
|            Kinyarwanda (kin)           |  0.953125 | 0.938462 | 0.945736 |
|              Kirghiz (kir)             |  0.927632 | 0.881250 | 0.903846 |
|           Komi-Permyak (koi)           |  0.750000 | 0.776786 | 0.763158 |
|              Konkani (kok)             |  0.893491 | 0.872832 | 0.883041 |
|               Komi (kom)               |  0.734177 | 0.690476 | 0.711656 |
|              Korean (kor)              |  0.989899 | 0.989899 | 0.989899 |
|          Karachay-Balkar (krc)         |  0.928571 | 0.917647 | 0.923077 |
|            Ripuarisch (ksh)            |  0.915789 | 0.896907 | 0.906250 |
|              Kurdish (kur)             |  0.977528 | 0.935484 | 0.956044 |
|              Ladino (lad)              |  0.985075 | 0.904110 | 0.942857 |
|                Lao (lao)               |  0.896552 | 0.812500 | 0.852459 |
|               Latin (lat)              |  0.741935 | 0.831325 | 0.784091 |
|              Latvian (lav)             |  0.710526 | 0.878049 | 0.785455 |
|             Lezghian (lez)             |  0.975309 | 0.877778 | 0.923977 |
|             Ligurian (lij)             |  0.951807 | 0.897727 | 0.923977 |
|             Limburgan (lim)            |  0.909091 | 0.921053 | 0.915033 |
|              Lingala (lin)             |  0.942857 | 0.814815 | 0.874172 |
|            Lithuanian (lit)            |  0.892857 | 0.925926 | 0.909091 |
|              Lombard (lmo)             |  0.766234 | 0.951613 | 0.848921 |
|           Northern Luri (lrc)          |  0.972222 | 0.875000 | 0.921053 |
|             Latgalian (ltg)            |  0.895349 | 0.865169 | 0.880000 |
|           Luxembourgish (ltz)          |  0.882353 | 0.750000 | 0.810811 |
|              Luganda (lug)             |  0.946429 | 0.883333 | 0.913793 |
|         Literary Chinese (lzh)         |  1.000000 | 1.000000 | 1.000000 |
|             Maithili (mai)             |  0.893617 | 0.823529 | 0.857143 |
|             Malayalam (mal)            |  1.000000 | 0.975000 | 0.987342 |
|          Banyumasan (map-bms)          |  0.924242 | 0.772152 | 0.841379 |
|              Marathi (mar)             |  0.874126 | 0.919118 | 0.896057 |
|              Moksha (mdf)              |  0.771242 | 0.830986 | 0.800000 |
|           Eastern Mari (mhr)           |  0.820000 | 0.860140 | 0.839590 |
|            Minangkabau (min)           |  0.973684 | 0.973684 | 0.973684 |
|            Macedonian (mkd)            |  0.895652 | 0.953704 | 0.923767 |
|             Malagasy (mlg)             |  1.000000 | 0.966102 | 0.982759 |
|              Maltese (mlt)             |  0.987952 | 0.964706 | 0.976190 |
|          Min Nan Chinese (nan)         |  0.975000 | 1.000000 | 0.987342 |
|             Mongolian (mon)            |  0.954545 | 0.933333 | 0.943820 |
|               Maori (mri)              |  0.985294 | 1.000000 | 0.992593 |
|           Western Mari (mrj)           |  0.966292 | 0.914894 | 0.939891 |
|               Malay (msa)              |  0.770270 | 0.695122 | 0.730769 |
|             Mirandese (mwl)            |  0.970588 | 0.891892 | 0.929577 |
|              Burmese (mya)             |  1.000000 | 0.964286 | 0.981818 |
|               Erzya (myv)              |  0.535714 | 0.681818 | 0.600000 |
|            Mazanderani (mzn)           |  0.968750 | 0.898551 | 0.932331 |
|            Neapolitan (nap)            |  0.892308 | 0.865672 | 0.878788 |
|              Navajo (nav)              |  0.984375 | 0.984375 | 0.984375 |
|         Classical Nahuatl (nci)        |  0.901408 | 0.761905 | 0.825806 |
|            Low German (nds)            |  0.896226 | 0.913462 | 0.904762 |
|        West Low German (nds-nl)        |  0.873563 | 0.835165 | 0.853933 |
|      Nepali (macrolanguage) (nep)      |  0.704545 | 0.861111 | 0.775000 |
|              Newari (new)              |  0.920000 | 0.741935 | 0.821429 |
|               Dutch (nld)              |  0.925926 | 0.872093 | 0.898204 |
|         Norwegian Nynorsk (nno)        |  0.847059 | 0.808989 | 0.827586 |
|              Bokmål (nob)              |  0.861386 | 0.852941 | 0.857143 |
|               Narom (nrm)              |  0.966667 | 0.983051 | 0.974790 |
|          Northern Sotho (nso)          |  0.897436 | 0.921053 | 0.909091 |
|              Occitan (oci)             |  0.958333 | 0.696970 | 0.807018 |
|          Livvi-Karelian (olo)          |  0.967742 | 0.937500 | 0.952381 |
|               Oriya (ori)              |  0.933333 | 1.000000 | 0.965517 |
|               Oromo (orm)              |  0.977528 | 0.915789 | 0.945652 |
|             Ossetian (oss)             |  0.958333 | 0.841463 | 0.896104 |
|            Pangasinan (pag)            |  0.847328 | 0.909836 | 0.877470 |
|             Pampanga (pam)             |  0.969697 | 0.780488 | 0.864865 |
|              Panjabi (pan)             |  1.000000 | 1.000000 | 1.000000 |
|            Papiamento (pap)            |  0.876190 | 0.920000 | 0.897561 |
|              Picard (pcd)              |  0.707317 | 0.568627 | 0.630435 |
|        Pennsylvania German (pdc)       |  0.827273 | 0.827273 | 0.827273 |
|          Palatine German (pfl)         |  0.882353 | 0.914634 | 0.898204 |
|          Western Panjabi (pnb)         |  0.964286 | 0.931034 | 0.947368 |
|              Polish (pol)              |  0.859813 | 0.910891 | 0.884615 |
|            Portuguese (por)            |  0.535714 | 0.833333 | 0.652174 |
|              Pushto (pus)              |  0.989362 | 0.902913 | 0.944162 |
|              Quechua (que)             |  0.979167 | 0.903846 | 0.940000 |
|      Tarantino dialect (roa-tara)      |  0.964912 | 0.901639 | 0.932203 |
|              Romansh (roh)             |  0.914894 | 0.895833 | 0.905263 |
|             Romanian (ron)             |  0.880597 | 0.880597 | 0.880597 |
|               Rusyn (rue)              |  0.932584 | 0.805825 | 0.864583 |
|             Aromanian (rup)            |  0.783333 | 0.758065 | 0.770492 |
|              Russian (rus)             |  0.517986 | 0.765957 | 0.618026 |
|               Yakut (sah)              |  0.954023 | 0.922222 | 0.937853 |
|             Sanskrit (san)             |  0.866667 | 0.951220 | 0.906977 |
|             Sicilian (scn)             |  0.984375 | 0.940299 | 0.961832 |
|               Scots (sco)              |  0.851351 | 0.900000 | 0.875000 |
|            Samogitian (sgs)            |  0.977011 | 0.876289 | 0.923913 |
|              Sinhala (sin)             |  0.406154 | 0.985075 | 0.575163 |
|              Slovak (slk)              |  0.956989 | 0.872549 | 0.912821 |
|              Slovene (slv)             |  0.907216 | 0.854369 | 0.880000 |
|           Northern Sami (sme)          |  0.949367 | 0.892857 | 0.920245 |
|               Shona (sna)              |  0.936508 | 0.855072 | 0.893939 |
|              Sindhi (snd)              |  0.984962 | 0.992424 | 0.988679 |
|              Somali (som)              |  0.949153 | 0.848485 | 0.896000 |
|              Spanish (spa)             |  0.584158 | 0.746835 | 0.655556 |
|             Albanian (sqi)             |  0.988095 | 0.912088 | 0.948571 |
|             Sardinian (srd)            |  0.957746 | 0.931507 | 0.944444 |
|              Sranan (srn)              |  0.985714 | 0.945205 | 0.965035 |
|              Serbian (srp)             |  0.950980 | 0.889908 | 0.919431 |
|          Saterfriesisch (stq)          |  0.962500 | 0.875000 | 0.916667 |
|             Sundanese (sun)            |  0.778846 | 0.910112 | 0.839378 |
|      Swahili (macrolanguage) (swa)     |  0.915493 | 0.878378 | 0.896552 |
|              Swedish (swe)             |  0.989247 | 0.958333 | 0.973545 |
|             Silesian (szl)             |  0.944444 | 0.904255 | 0.923913 |
|               Tamil (tam)              |  0.990000 | 0.970588 | 0.980198 |
|               Tatar (tat)              |  0.942029 | 0.902778 | 0.921986 |
|               Tulu (tcy)               |  0.980519 | 0.967949 | 0.974194 |
|              Telugu (tel)              |  0.965986 | 0.965986 | 0.965986 |
|               Tetum (tet)              |  0.898734 | 0.855422 | 0.876543 |
|               Tajik (tgk)              |  0.974684 | 0.939024 | 0.956522 |
|              Tagalog (tgl)             |  0.965909 | 0.934066 | 0.949721 |
|               Thai (tha)               |  0.923077 | 0.882353 | 0.902256 |
|              Tongan (ton)              |  0.970149 | 0.890411 | 0.928571 |
|              Tswana (tsn)              |  0.888889 | 0.926316 | 0.907216 |
|              Turkmen (tuk)             |  0.968000 | 0.889706 | 0.927203 |
|              Turkish (tur)             |  0.871287 | 0.926316 | 0.897959 |
|               Tuvan (tyv)              |  0.948454 | 0.859813 | 0.901961 |
|              Udmurt (udm)              |  0.989362 | 0.894231 | 0.939394 |
|              Uighur (uig)              |  1.000000 | 0.953333 | 0.976109 |
|             Ukrainian (ukr)            |  0.893617 | 0.875000 | 0.884211 |
|               Urdu (urd)               |  1.000000 | 1.000000 | 1.000000 |
|               Uzbek (uzb)              |  0.636042 | 0.886700 | 0.740741 |
|             Venetian (vec)             |  1.000000 | 0.941176 | 0.969697 |
|               Veps (vep)               |  0.858586 | 0.965909 | 0.909091 |
|            Vietnamese (vie)            |  1.000000 | 0.940476 | 0.969325 |
|              Vlaams (vls)              |  0.885714 | 0.898551 | 0.892086 |
|              Volapük (vol)             |  0.975309 | 0.975309 | 0.975309 |
|               Võro (vro)               |  0.855670 | 0.864583 | 0.860104 |
|               Waray (war)              |  0.972222 | 0.909091 | 0.939597 |
|              Walloon (wln)             |  0.742138 | 0.893939 | 0.810997 |
|               Wolof (wol)              |  0.882979 | 0.954023 | 0.917127 |
|            Wu Chinese (wuu)            |  0.961538 | 0.833333 | 0.892857 |
|               Xhosa (xho)              |  0.934066 | 0.867347 | 0.899471 |
|            Mingrelian (xmf)            |  0.958333 | 0.929293 | 0.943590 |
|              Yiddish (yid)             |  0.984375 | 0.875000 | 0.926471 |
|              Yoruba (yor)              |  0.868421 | 0.857143 | 0.862745 |
|              Zeeuws (zea)              |  0.879518 | 0.793478 | 0.834286 |
|           Cantonese (zh-yue)           |  0.896552 | 0.812500 | 0.852459 |
|         Standard Chinese (zho)         |  0.906250 | 0.935484 | 0.920635 |
|                accuracy                |  0.881051 | 0.881051 | 0.881051 |
|                macro avg               |  0.903245 | 0.880618 | 0.888996 |
|              weighted avg              |  0.894174 | 0.881051 | 0.884520 |

</details>

<details>
<summary>Token level</summary>

|                language                | precision |  recall  | f1-score |
|:--------------------------------------:|:---------:|:--------:|:--------:|
|             Achinese (ace)             |  0.873846 | 0.827988 | 0.850299 |
|             Afrikaans (afr)            |  0.638060 | 0.732334 | 0.681954 |
|         Alemannic German (als)         |  0.673780 | 0.547030 | 0.603825 |
|              Amharic (amh)             |  0.997743 | 0.954644 | 0.975717 |
|            Old English (ang)           |  0.840816 | 0.693603 | 0.760148 |
|              Arabic (ara)              |  0.768737 | 0.840749 | 0.803132 |
|             Aragonese (arg)            |  0.493671 | 0.505181 | 0.499360 |
|          Egyptian Arabic (arz)         |  0.823529 | 0.741935 | 0.780606 |
|             Assamese (asm)             |  0.948454 | 0.893204 | 0.920000 |
|             Asturian (ast)             |  0.490000 | 0.508299 | 0.498982 |
|               Avar (ava)               |  0.813636 | 0.655678 | 0.726166 |
|              Aymara (aym)              |  0.795833 | 0.779592 | 0.787629 |
|         South Azerbaijani (azb)        |  0.832836 | 0.863777 | 0.848024 |
|            Azerbaijani (aze)           |  0.867470 | 0.800000 | 0.832370 |
|              Bashkir (bak)             |  0.851852 | 0.750000 | 0.797688 |
|             Bavarian (bar)             |  0.560897 | 0.522388 | 0.540958 |
|           Central Bikol (bcl)          |  0.708229 | 0.668235 | 0.687651 |
| Belarusian (Taraschkewiza) (be-tarask) |  0.615635 | 0.526462 | 0.567568 |
|            Belarusian (bel)            |  0.539952 | 0.597855 | 0.567430 |
|              Bengali (ben)             |  0.830275 | 0.885086 | 0.856805 |
|             Bhojpuri (bho)             |  0.723118 | 0.691517 | 0.706965 |
|              Banjar (bjn)              |  0.619586 | 0.726269 | 0.668699 |
|              Tibetan (bod)             |  0.999537 | 0.991728 | 0.995617 |
|              Bosnian (bos)             |  0.330849 | 0.403636 | 0.363636 |
|            Bishnupriya (bpy)           |  0.941634 | 0.949020 | 0.945312 |
|              Breton (bre)              |  0.772222 | 0.745308 | 0.758527 |
|             Bulgarian (bul)            |  0.771505 | 0.706897 | 0.737789 |
|              Buryat (bxr)              |  0.741935 | 0.753149 | 0.747500 |
|              Catalan (cat)             |  0.528716 | 0.610136 | 0.566516 |
|             Chavacano (cbk)            |  0.409449 | 0.312625 | 0.354545 |
|             Min Dong (cdo)             |  0.951264 | 0.936057 | 0.943599 |
|              Cebuano (ceb)             |  0.888298 | 0.876640 | 0.882431 |
|               Czech (ces)              |  0.806045 | 0.758294 | 0.781441 |
|              Chechen (che)             |  0.857143 | 0.600000 | 0.705882 |
|             Cherokee (chr)             |  0.997840 | 0.952577 | 0.974684 |
|              Chuvash (chv)             |  0.874346 | 0.776744 | 0.822660 |
|          Central Kurdish (ckb)         |  0.984848 | 0.953545 | 0.968944 |
|              Cornish (cor)             |  0.747596 | 0.807792 | 0.776529 |
|             Corsican (cos)             |  0.673913 | 0.708571 | 0.690808 |
|           Crimean Tatar (crh)          |  0.498801 | 0.700337 | 0.582633 |
|             Kashubian (csb)            |  0.797059 | 0.794721 | 0.795888 |
|               Welsh (cym)              |  0.829609 | 0.841360 | 0.835443 |
|              Danish (dan)              |  0.649789 | 0.622222 | 0.635707 |
|              German (deu)              |  0.559406 | 0.763514 | 0.645714 |
|               Dimli (diq)              |  0.835580 | 0.763547 | 0.797941 |
|              Dhivehi (div)             |  1.000000 | 0.980645 | 0.990228 |
|           Lower Sorbian (dsb)          |  0.740484 | 0.694805 | 0.716918 |
|              Doteli (dty)              |  0.616314 | 0.527132 | 0.568245 |
|              Emilian (egl)             |  0.822993 | 0.769625 | 0.795414 |
|           Modern Greek (ell)           |  0.972043 | 0.963753 | 0.967880 |
|              English (eng)             |  0.260492 | 0.724346 | 0.383183 |
|             Esperanto (epo)            |  0.766764 | 0.716621 | 0.740845 |
|             Estonian (est)             |  0.698885 | 0.673835 | 0.686131 |
|              Basque (eus)              |  0.882716 | 0.841176 | 0.861446 |
|           Extremaduran (ext)           |  0.570605 | 0.511628 | 0.539510 |
|              Faroese (fao)             |  0.773987 | 0.784017 | 0.778970 |
|              Persian (fas)             |  0.709836 | 0.809346 | 0.756332 |
|              Finnish (fin)             |  0.866261 | 0.796089 | 0.829694 |
|              French (fra)              |  0.496263 | 0.700422 | 0.580927 |
|              Arpitan (frp)             |  0.663366 | 0.584302 | 0.621329 |
|          Western Frisian (fry)         |  0.750000 | 0.756148 | 0.753061 |
|             Friulian (fur)             |  0.713555 | 0.675545 | 0.694030 |
|              Gagauz (gag)              |  0.728125 | 0.677326 | 0.701807 |
|          Scottish Gaelic (gla)         |  0.831601 | 0.817996 | 0.824742 |
|               Irish (gle)              |  0.868852 | 0.801296 | 0.833708 |
|             Galician (glg)             |  0.469816 | 0.454315 | 0.461935 |
|              Gilaki (glk)              |  0.703883 | 0.687204 | 0.695444 |
|               Manx (glv)               |  0.873047 | 0.886905 | 0.879921 |
|              Guarani (grn)             |  0.848580 | 0.793510 | 0.820122 |
|             Gujarati (guj)             |  0.995643 | 0.926978 | 0.960084 |
|           Hakka Chinese (hak)          |  0.898403 | 0.904971 | 0.901675 |
|          Haitian Creole (hat)          |  0.719298 | 0.518987 | 0.602941 |
|               Hausa (hau)              |  0.815353 | 0.829114 | 0.822176 |
|          Serbo-Croatian (hbs)          |  0.343465 | 0.244589 | 0.285714 |
|              Hebrew (heb)              |  0.891304 | 0.933941 | 0.912125 |
|            Fiji Hindi (hif)            |  0.662577 | 0.664615 | 0.663594 |
|               Hindi (hin)              |  0.782301 | 0.778169 | 0.780229 |
|             Croatian (hrv)             |  0.360308 | 0.374000 | 0.367026 |
|           Upper Sorbian (hsb)          |  0.745763 | 0.611111 | 0.671756 |
|             Hungarian (hun)            |  0.876812 | 0.846154 | 0.861210 |
|             Armenian (hye)             |  0.988201 | 0.917808 | 0.951705 |
|               Igbo (ibo)               |  0.825397 | 0.696429 | 0.755448 |
|                Ido (ido)               |  0.760479 | 0.814103 | 0.786378 |
|            Interlingue (ile)           |  0.701299 | 0.580645 | 0.635294 |
|               Iloko (ilo)              |  0.688356 | 0.844538 | 0.758491 |
|            Interlingua (ina)           |  0.577889 | 0.588235 | 0.583016 |
|            Indonesian (ind)            |  0.415879 | 0.514019 | 0.459770 |
|             Icelandic (isl)            |  0.855263 | 0.790754 | 0.821745 |
|              Italian (ita)             |  0.474576 | 0.561247 | 0.514286 |
|          Jamaican Patois (jam)         |  0.826087 | 0.791667 | 0.808511 |
|             Javanese (jav)             |  0.670130 | 0.658163 | 0.664093 |
|              Lojban (jbo)              |  0.896861 | 0.917431 | 0.907029 |
|             Japanese (jpn)             |  0.931373 | 0.848214 | 0.887850 |
|            Karakalpak (kaa)            |  0.790393 | 0.827744 | 0.808637 |
|              Kabyle (kab)              |  0.828571 | 0.759162 | 0.792350 |
|              Kannada (kan)             |  0.879357 | 0.847545 | 0.863158 |
|             Georgian (kat)             |  0.916399 | 0.907643 | 0.912000 |
|              Kazakh (kaz)              |  0.900901 | 0.819672 | 0.858369 |
|             Kabardian (kbd)            |  0.923345 | 0.892256 | 0.907534 |
|           Central Khmer (khm)          |  0.976667 | 0.816156 | 0.889226 |
|            Kinyarwanda (kin)           |  0.824324 | 0.726190 | 0.772152 |
|              Kirghiz (kir)             |  0.674766 | 0.779698 | 0.723447 |
|           Komi-Permyak (koi)           |  0.652830 | 0.633700 | 0.643123 |
|              Konkani (kok)             |  0.778865 | 0.728938 | 0.753075 |
|               Komi (kom)               |  0.737374 | 0.572549 | 0.644592 |
|              Korean (kor)              |  0.984615 | 0.967603 | 0.976035 |
|          Karachay-Balkar (krc)         |  0.869416 | 0.857627 | 0.863481 |
|            Ripuarisch (ksh)            |  0.709859 | 0.649485 | 0.678331 |
|              Kurdish (kur)             |  0.883777 | 0.862884 | 0.873206 |
|              Ladino (lad)              |  0.660920 | 0.576441 | 0.615797 |
|                Lao (lao)               |  0.986175 | 0.918455 | 0.951111 |
|               Latin (lat)              |  0.581250 | 0.636986 | 0.607843 |
|              Latvian (lav)             |  0.824513 | 0.797844 | 0.810959 |
|             Lezghian (lez)             |  0.898955 | 0.793846 | 0.843137 |
|             Ligurian (lij)             |  0.662903 | 0.677100 | 0.669927 |
|             Limburgan (lim)            |  0.615385 | 0.581818 | 0.598131 |
|              Lingala (lin)             |  0.836207 | 0.763780 | 0.798354 |
|            Lithuanian (lit)            |  0.756329 | 0.804714 | 0.779772 |
|              Lombard (lmo)             |  0.556818 | 0.536986 | 0.546722 |
|           Northern Luri (lrc)          |  0.838574 | 0.753296 | 0.793651 |
|             Latgalian (ltg)            |  0.759531 | 0.755102 | 0.757310 |
|           Luxembourgish (ltz)          |  0.645062 | 0.614706 | 0.629518 |
|              Luganda (lug)             |  0.787535 | 0.805797 | 0.796562 |
|         Literary Chinese (lzh)         |  0.921951 | 0.949749 | 0.935644 |
|             Maithili (mai)             |  0.777778 | 0.761658 | 0.769634 |
|             Malayalam (mal)            |  0.993377 | 0.949367 | 0.970874 |
|          Banyumasan (map-bms)          |  0.531429 | 0.453659 | 0.489474 |
|              Marathi (mar)             |  0.748744 | 0.818681 | 0.782152 |
|              Moksha (mdf)              |  0.728745 | 0.800000 | 0.762712 |
|           Eastern Mari (mhr)           |  0.790323 | 0.760870 | 0.775316 |
|            Minangkabau (min)           |  0.953271 | 0.886957 | 0.918919 |
|            Macedonian (mkd)            |  0.816399 | 0.849722 | 0.832727 |
|             Malagasy (mlg)             |  0.925187 | 0.918317 | 0.921739 |
|              Maltese (mlt)             |  0.869421 | 0.890017 | 0.879599 |
|          Min Nan Chinese (nan)         |  0.743707 | 0.820707 | 0.780312 |
|             Mongolian (mon)            |  0.852194 | 0.838636 | 0.845361 |
|               Maori (mri)              |  0.934726 | 0.937173 | 0.935948 |
|           Western Mari (mrj)           |  0.818792 | 0.827119 | 0.822934 |
|               Malay (msa)              |  0.508065 | 0.376119 | 0.432247 |
|             Mirandese (mwl)            |  0.650407 | 0.685225 | 0.667362 |
|              Burmese (mya)             |  0.995968 | 0.972441 | 0.984064 |
|               Erzya (myv)              |  0.475783 | 0.503012 | 0.489019 |
|            Mazanderani (mzn)           |  0.775362 | 0.701639 | 0.736661 |
|            Neapolitan (nap)            |  0.628993 | 0.595349 | 0.611708 |
|              Navajo (nav)              |  0.955882 | 0.937500 | 0.946602 |
|         Classical Nahuatl (nci)        |  0.679758 | 0.589005 | 0.631136 |
|            Low German (nds)            |  0.669789 | 0.690821 | 0.680143 |
|        West Low German (nds-nl)        |  0.513889 | 0.504545 | 0.509174 |
|      Nepali (macrolanguage) (nep)      |  0.640476 | 0.649758 | 0.645084 |
|              Newari (new)              |  0.928571 | 0.745902 | 0.827273 |
|               Dutch (nld)              |  0.553763 | 0.553763 | 0.553763 |
|         Norwegian Nynorsk (nno)        |  0.569277 | 0.519231 | 0.543103 |
|              Bokmål (nob)              |  0.519856 | 0.562500 | 0.540338 |
|               Narom (nrm)              |  0.691275 | 0.605882 | 0.645768 |
|          Northern Sotho (nso)          |  0.950276 | 0.815166 | 0.877551 |
|              Occitan (oci)             |  0.483444 | 0.366834 | 0.417143 |
|          Livvi-Karelian (olo)          |  0.816850 | 0.790780 | 0.803604 |
|               Oriya (ori)              |  0.981481 | 0.963636 | 0.972477 |
|               Oromo (orm)              |  0.885714 | 0.829218 | 0.856536 |
|             Ossetian (oss)             |  0.822006 | 0.855219 | 0.838284 |
|            Pangasinan (pag)            |  0.842105 | 0.715655 | 0.773748 |
|             Pampanga (pam)             |  0.770000 | 0.435028 | 0.555957 |
|              Panjabi (pan)             |  0.996154 | 0.984791 | 0.990440 |
|            Papiamento (pap)            |  0.674672 | 0.661670 | 0.668108 |
|              Picard (pcd)              |  0.407895 | 0.356322 | 0.380368 |
|        Pennsylvania German (pdc)       |  0.487047 | 0.509485 | 0.498013 |
|          Palatine German (pfl)         |  0.614173 | 0.570732 | 0.591656 |
|          Western Panjabi (pnb)         |  0.926267 | 0.887417 | 0.906426 |
|              Polish (pol)              |  0.797059 | 0.734417 | 0.764457 |
|            Portuguese (por)            |  0.500914 | 0.586724 | 0.540434 |
|              Pushto (pus)              |  0.941489 | 0.898477 | 0.919481 |
|              Quechua (que)             |  0.854167 | 0.797665 | 0.824950 |
|      Tarantino dialect (roa-tara)      |  0.669794 | 0.724138 | 0.695906 |
|              Romansh (roh)             |  0.745527 | 0.760649 | 0.753012 |
|             Romanian (ron)             |  0.805486 | 0.769048 | 0.786845 |
|               Rusyn (rue)              |  0.718543 | 0.645833 | 0.680251 |
|             Aromanian (rup)            |  0.288482 | 0.730245 | 0.413580 |
|              Russian (rus)             |  0.530120 | 0.690583 | 0.599805 |
|               Yakut (sah)              |  0.853521 | 0.865714 | 0.859574 |
|             Sanskrit (san)             |  0.931343 | 0.896552 | 0.913616 |
|             Sicilian (scn)             |  0.734139 | 0.618321 | 0.671271 |
|               Scots (sco)              |  0.571429 | 0.540816 | 0.555701 |
|            Samogitian (sgs)            |  0.829167 | 0.748120 | 0.786561 |
|              Sinhala (sin)             |  0.909474 | 0.935065 | 0.922092 |
|              Slovak (slk)              |  0.738235 | 0.665782 | 0.700139 |
|              Slovene (slv)             |  0.671123 | 0.662269 | 0.666667 |
|           Northern Sami (sme)          |  0.800676 | 0.825784 | 0.813036 |
|               Shona (sna)              |  0.761702 | 0.724696 | 0.742739 |
|              Sindhi (snd)              |  0.950172 | 0.946918 | 0.948542 |
|              Somali (som)              |  0.849462 | 0.802030 | 0.825065 |
|              Spanish (spa)             |  0.325234 | 0.413302 | 0.364017 |
|             Albanian (sqi)             |  0.875899 | 0.832479 | 0.853637 |
|             Sardinian (srd)            |  0.750000 | 0.711061 | 0.730012 |
|              Sranan (srn)              |  0.888889 | 0.771084 | 0.825806 |
|              Serbian (srp)             |  0.824561 | 0.814356 | 0.819427 |
|          Saterfriesisch (stq)          |  0.790087 | 0.734417 | 0.761236 |
|             Sundanese (sun)            |  0.764192 | 0.631769 | 0.691700 |
|      Swahili (macrolanguage) (swa)     |  0.763496 | 0.796247 | 0.779528 |
|              Swedish (swe)             |  0.838284 | 0.723647 | 0.776758 |
|             Silesian (szl)             |  0.819788 | 0.750809 | 0.783784 |
|               Tamil (tam)              |  0.985765 | 0.955172 | 0.970228 |
|               Tatar (tat)              |  0.469780 | 0.795349 | 0.590674 |
|               Tulu (tcy)               |  0.893300 | 0.873786 | 0.883436 |
|              Telugu (tel)              |  1.000000 | 0.913690 | 0.954899 |
|               Tetum (tet)              |  0.765116 | 0.744344 | 0.754587 |
|               Tajik (tgk)              |  0.828418 | 0.813158 | 0.820717 |
|              Tagalog (tgl)             |  0.751468 | 0.757396 | 0.754420 |
|               Thai (tha)               |  0.933884 | 0.807143 | 0.865900 |
|              Tongan (ton)              |  0.920245 | 0.923077 | 0.921659 |
|              Tswana (tsn)              |  0.873397 | 0.889070 | 0.881164 |
|              Turkmen (tuk)             |  0.898438 | 0.837887 | 0.867107 |
|              Turkish (tur)             |  0.666667 | 0.716981 | 0.690909 |
|               Tuvan (tyv)              |  0.857143 | 0.805063 | 0.830287 |
|              Udmurt (udm)              |  0.865517 | 0.756024 | 0.807074 |
|              Uighur (uig)              |  0.991597 | 0.967213 | 0.979253 |
|             Ukrainian (ukr)            |  0.771341 | 0.702778 | 0.735465 |
|               Urdu (urd)               |  0.877647 | 0.855505 | 0.866434 |
|               Uzbek (uzb)              |  0.655652 | 0.797040 | 0.719466 |
|             Venetian (vec)             |  0.611111 | 0.527233 | 0.566082 |
|               Veps (vep)               |  0.672862 | 0.688213 | 0.680451 |
|            Vietnamese (vie)            |  0.932406 | 0.914230 | 0.923228 |
|              Vlaams (vls)              |  0.594427 | 0.501305 | 0.543909 |
|              Volapük (vol)             |  0.765625 | 0.942308 | 0.844828 |
|               Võro (vro)               |  0.797203 | 0.740260 | 0.767677 |
|               Waray (war)              |  0.930876 | 0.930876 | 0.930876 |
|              Walloon (wln)             |  0.636804 | 0.693931 | 0.664141 |
|               Wolof (wol)              |  0.864220 | 0.845601 | 0.854809 |
|            Wu Chinese (wuu)            |  0.848921 | 0.830986 | 0.839858 |
|               Xhosa (xho)              |  0.837398 | 0.759214 | 0.796392 |
|            Mingrelian (xmf)            |  0.943396 | 0.874126 | 0.907441 |
|              Yiddish (yid)             |  0.955729 | 0.897311 | 0.925599 |
|              Yoruba (yor)              |  0.812010 | 0.719907 | 0.763190 |
|              Zeeuws (zea)              |  0.617737 | 0.550409 | 0.582133 |
|           Cantonese (zh-yue)           |  0.859649 | 0.649007 | 0.739623 |
|         Standard Chinese (zho)         |  0.845528 | 0.781955 | 0.812500 |
|                accuracy                |  0.749527 | 0.749527 | 0.749527 |
|                macro avg               |  0.762866 | 0.742101 | 0.749261 |
|              weighted avg              |  0.762006 | 0.749527 | 0.752910 |

</details>

As can be seen, the model outperforms on groups of similar or dialects languages. 
For instance, the f1 scores for the Persian language and similar languages like Gilaki, Northern Luri,  Central Kurdish, Kurdish, and Mazanderani are 92%, 91%, 92%, 99%, 94%, and 93%, respectively. 

# How to Install
We recommend Python 3.7 or higher, PyTorch 1.6.0 or higher. The code does not work with Python 2.7.

```bash
pip install git+https://github.com/m3hrdadfi/zabanshenas.git
```

## How to Use

You can use this code snippet to identify the most likely language of a written document. You just have to say: ZABANSHENAS (detector) -> BESHNAS (detect) 😎. 

*Sounds interesting, doesn't it?*

```python
from zabanshenas.zabanshenas import Zabanshenas

zabanshenas = Zabanshenas()
text = "زیر لکه‌های زمان احساسات محو میشن تو یکی دیگه شدی و من هنوز اینجام"
# Beneath the strains of time, the feelings disappear, you are someone else, I'm still right here!

r = zabanshenas.detect(text, return_all_scores=False)
print(r)
```

Output:
```text
[
  {
    "language": "Persian",
    "code": "fas",
    "score": 0.6105580925941467
  }
]
```

Or you can find out all the candidates' scores using the following snippet.

```python
from zabanshenas.zabanshenas import Zabanshenas

zabanshenas = Zabanshenas()
text = "زیر لکه‌های زمان احساسات محو میشن تو یکی دیگه شدی و من هنوز اینجام"
# Beneath the strains of time, the feelings disappear, you are someone else, I'm still right here!

r = zabanshenas.detect(text, return_all_scores=True)
print(r)
```
Output:

</details>

<details>
<summary>See all the 235 candidates</summary>

```text
[
  [
    {
      "language": "Persian",
      "code": "fas",
      "score": 0.6105580925941467
    },
    {
      "language": "Gilaki",
      "code": "glk",
      "score": 0.29982829093933105
    },
    {
      "language": "Northern Luri",
      "code": "lrc",
      "score": 0.04840774089097977
    },
    {
      "language": "Mazanderani",
      "code": "mzn",
      "score": 0.030142733827233315
    },
    {
      "language": "South Azerbaijani",
      "code": "azb",
      "score": 0.005220199003815651
    },
    {
      "language": "Urdu",
      "code": "urd",
      "score": 0.0019745035097002983
    },
    {
      "language": "Pushto",
      "code": "pus",
      "score": 0.0015690263826400042
    },
    {
      "language": "Western Panjabi",
      "code": "pnb",
      "score": 0.0005721596535295248
    },
    {
      "language": "Central Kurdish",
      "code": "ckb",
      "score": 0.00025537016335874796
    },
    {
      "language": "Sindhi",
      "code": "snd",
      "score": 0.0001820324978325516
    },
    {
      "language": "Egyptian Arabic",
      "code": "arz",
      "score": 0.0001247940381290391
    },
    {
      "language": "Arabic",
      "code": "ara",
      "score": 7.754910620860755e-05
    },
    {
      "language": "Korean",
      "code": "kor",
      "score": 5.718228203477338e-05
    },
    {
      "language": "Fiji Hindi",
      "code": "hif",
      "score": 3.5903740354115143e-05
    },
    {
      "language": "Uighur",
      "code": "uig",
      "score": 3.5565532016335055e-05
    },
    {
      "language": "Maori",
      "code": "mri",
      "score": 2.1078320060041733e-05
    },
    {
      "language": "Literary Chinese",
      "code": "lzh",
      "score": 2.09943773370469e-05
    },
    {
      "language": "Navajo",
      "code": "nav",
      "score": 1.8877935872296803e-05
    },
    {
      "language": "Mongolian",
      "code": "mon",
      "score": 1.783044899639208e-05
    },
    {
      "language": "Basque",
      "code": "eus",
      "score": 1.2980432074982673e-05
    },
    {
      "language": "Moksha",
      "code": "mdf",
      "score": 1.2325609532126691e-05
    },
    {
      "language": "Tongan",
      "code": "ton",
      "score": 1.1610675755946431e-05
    },
    {
      "language": "Min Dong",
      "code": "cdo",
      "score": 1.1508132956805639e-05
    },
    {
      "language": "Sinhala",
      "code": "sin",
      "score": 1.0617596672091167e-05
    },
    {
      "language": "Venetian",
      "code": "vec",
      "score": 1.0375520105299074e-05
    },
    {
      "language": "Western Mari",
      "code": "mrj",
      "score": 1.0316403859178536e-05
    },
    {
      "language": "Malayalam",
      "code": "mal",
      "score": 1.0265099263051525e-05
    },
    {
      "language": "Interlingua",
      "code": "ina",
      "score": 1.0040446795755997e-05
    },
    {
      "language": "Tatar",
      "code": "tat",
      "score": 9.836200661084149e-06
    },
    {
      "language": "Cantonese",
      "code": "zh-yue",
      "score": 9.80662207439309e-06
    },
    {
      "language": "Wu Chinese",
      "code": "wuu",
      "score": 9.661145668360405e-06
    },
    {
      "language": "Igbo",
      "code": "ibo",
      "score": 9.207592484017368e-06
    },
    {
      "language": "Waray",
      "code": "war",
      "score": 8.970115231932141e-06
    },
    {
      "language": "Yiddish",
      "code": "yid",
      "score": 8.926748705562204e-06
    },
    {
      "language": "Udmurt",
      "code": "udm",
      "score": 8.702583727426827e-06
    },
    {
      "language": "Dhivehi",
      "code": "div",
      "score": 8.36203707876848e-06
    },
    {
      "language": "Newari",
      "code": "new",
      "score": 8.140945283230394e-06
    },
    {
      "language": "Karachay-Balkar",
      "code": "krc",
      "score": 8.123539373627864e-06
    },
    {
      "language": "Lojban",
      "code": "jbo",
      "score": 8.114019692584407e-06
    },
    {
      "language": "Sanskrit",
      "code": "san",
      "score": 8.087784408417065e-06
    },
    {
      "language": "Luganda",
      "code": "lug",
      "score": 8.023569534998387e-06
    },
    {
      "language": "Maithili",
      "code": "mai",
      "score": 7.723083399469033e-06
    },
    {
      "language": "Kirghiz",
      "code": "kir",
      "score": 7.715119863860309e-06
    },
    {
      "language": "Standard Chinese",
      "code": "zho",
      "score": 7.5126054071006365e-06
    },
    {
      "language": "Amharic",
      "code": "amh",
      "score": 7.451813871739432e-06
    },
    {
      "language": "Chechen",
      "code": "che",
      "score": 7.444541097356705e-06
    },
    {
      "language": "Gujarati",
      "code": "guj",
      "score": 7.395997727144277e-06
    },
    {
      "language": "Tibetan",
      "code": "bod",
      "score": 7.390805421891855e-06
    },
    {
      "language": "Komi",
      "code": "kom",
      "score": 7.373077551164897e-06
    },
    {
      "language": "Lao",
      "code": "lao",
      "score": 7.351867679972202e-06
    },
    {
      "language": "Wolof",
      "code": "wol",
      "score": 7.305452982109273e-06
    },
    {
      "language": "Silesian",
      "code": "szl",
      "score": 7.301976893359097e-06
    },
    {
      "language": "Northern Sotho",
      "code": "nso",
      "score": 7.2927336987049785e-06
    },
    {
      "language": "Armenian",
      "code": "hye",
      "score": 7.243447726068553e-06
    },
    {
      "language": "Arpitan",
      "code": "frp",
      "score": 7.137540251278551e-06
    },
    {
      "language": "Bishnupriya",
      "code": "bpy",
      "score": 7.062033091642661e-06
    },
    {
      "language": "Azerbaijani",
      "code": "aze",
      "score": 6.906778253323864e-06
    },
    {
      "language": "Tajik",
      "code": "tgk",
      "score": 6.730050699843559e-06
    },
    {
      "language": "Old English ",
      "code": "ang",
      "score": 6.6442084971640725e-06
    },
    {
      "language": "Marathi",
      "code": "mar",
      "score": 6.63194168737391e-06
    },
    {
      "language": "Kurdish",
      "code": "kur",
      "score": 6.615779057028703e-06
    },
    {
      "language": "Lithuanian",
      "code": "lit",
      "score": 6.561998816323467e-06
    },
    {
      "language": "Russian",
      "code": "rus",
      "score": 6.4370215113740414e-06
    },
    {
      "language": "Tulu",
      "code": "tcy",
      "score": 6.370255960064242e-06
    },
    {
      "language": "Extremaduran",
      "code": "ext",
      "score": 6.3398160818906035e-06
    },
    {
      "language": "Aymara",
      "code": "aym",
      "score": 6.288398708420573e-06
    },
    {
      "language": "Lower Sorbian",
      "code": "dsb",
      "score": 6.209619641595054e-06
    },
    {
      "language": "Classical Nahuatl",
      "code": "nci",
      "score": 5.954705557087436e-06
    },
    {
      "language": "Polish",
      "code": "pol",
      "score": 5.952156243438367e-06
    },
    {
      "language": "Cebuano",
      "code": "ceb",
      "score": 5.911888820264721e-06
    },
    {
      "language": "Hakka Chinese",
      "code": "hak",
      "score": 5.756284735980444e-06
    },
    {
      "language": "Georgian",
      "code": "kat",
      "score": 5.656391749653267e-06
    },
    {
      "language": "Mingrelian",
      "code": "xmf",
      "score": 5.57373004994588e-06
    },
    {
      "language": "Telugu",
      "code": "tel",
      "score": 5.5334053286060225e-06
    },
    {
      "language": "Doteli",
      "code": "dty",
      "score": 5.510717073775595e-06
    },
    {
      "language": "Portuguese",
      "code": "por",
      "score": 5.50901131646242e-06
    },
    {
      "language": "Komi-Permyak",
      "code": "koi",
      "score": 5.447328476293478e-06
    },
    {
      "language": "Eastern Mari",
      "code": "mhr",
      "score": 5.414771294454113e-06
    },
    {
      "language": "Lezghian",
      "code": "lez",
      "score": 5.2741329454875086e-06
    },
    {
      "language": "Nepali (macrolanguage)",
      "code": "nep",
      "score": 5.273408532957546e-06
    },
    {
      "language": "Samogitian",
      "code": "sgs",
      "score": 5.207636149862083e-06
    },
    {
      "language": "Bhojpuri",
      "code": "bho",
      "score": 5.19551804245566e-06
    },
    {
      "language": "Occitan",
      "code": "oci",
      "score": 5.172901182959322e-06
    },
    {
      "language": "Western Frisian",
      "code": "fry",
      "score": 5.066170615464216e-06
    },
    {
      "language": "Vlaams",
      "code": "vls",
      "score": 5.014707312511746e-06
    },
    {
      "language": "Japanese",
      "code": "jpn",
      "score": 4.986791282135528e-06
    },
    {
      "language": "V\u00f5ro",
      "code": "vro",
      "score": 4.9785726332629565e-06
    },
    {
      "language": "Rusyn",
      "code": "rue",
      "score": 4.937043286190601e-06
    },
    {
      "language": "Hindi",
      "code": "hin",
      "score": 4.9325194595439825e-06
    },
    {
      "language": "Sicilian",
      "code": "scn",
      "score": 4.8434171731059905e-06
    },
    {
      "language": "Somali",
      "code": "som",
      "score": 4.722482117358595e-06
    },
    {
      "language": "Galician",
      "code": "glg",
      "score": 4.664954758482054e-06
    },
    {
      "language": "Kazakh",
      "code": "kaz",
      "score": 4.485120825847844e-06
    },
    {
      "language": "Kannada",
      "code": "kan",
      "score": 4.438274572748924e-06
    },
    {
      "language": "Oromo",
      "code": "orm",
      "score": 4.422903202794259e-06
    },
    {
      "language": "Albanian",
      "code": "sqi",
      "score": 4.410150268085999e-06
    },
    {
      "language": "Minangkabau",
      "code": "min",
      "score": 4.407007509144023e-06
    },
    {
      "language": "Finnish",
      "code": "fin",
      "score": 4.374884611024754e-06
    },
    {
      "language": "Ossetian",
      "code": "oss",
      "score": 4.322507265897002e-06
    },
    {
      "language": "Volap\u00fck",
      "code": "vol",
      "score": 4.30220188718522e-06
    },
    {
      "language": "Min Nan Chinese",
      "code": "nan",
      "score": 4.2357942220405675e-06
    },
    {
      "language": "Bashkir",
      "code": "bak",
      "score": 4.212616204313235e-06
    },
    {
      "language": "Ligurian",
      "code": "lij",
      "score": 4.1821313061518595e-06
    },
    {
      "language": "Welsh",
      "code": "cym",
      "score": 4.174029982095817e-06
    },
    {
      "language": "Slovene",
      "code": "slv",
      "score": 4.172954504610971e-06
    },
    {
      "language": "Dimli",
      "code": "diq",
      "score": 4.078176516486565e-06
    },
    {
      "language": "Chuvash",
      "code": "chv",
      "score": 4.048466053063748e-06
    },
    {
      "language": "Panjabi",
      "code": "pan",
      "score": 3.940522674383828e-06
    },
    {
      "language": "Cornish",
      "code": "cor",
      "score": 3.940297119697789e-06
    },
    {
      "language": "West Low German",
      "code": "nds-nl",
      "score": 3.926987574232044e-06
    },
    {
      "language": "Cherokee",
      "code": "chr",
      "score": 3.9112833292165305e-06
    },
    {
      "language": "Ido",
      "code": "ido",
      "score": 3.892145286954474e-06
    },
    {
      "language": "Friulian",
      "code": "fur",
      "score": 3.869370175380027e-06
    },
    {
      "language": "Ukrainian",
      "code": "ukr",
      "score": 3.7814761526533403e-06
    },
    {
      "language": "Vietnamese",
      "code": "vie",
      "score": 3.7795757634739857e-06
    },
    {
      "language": "Emilian",
      "code": "egl",
      "score": 3.7286854421836324e-06
    },
    {
      "language": "Hungarian",
      "code": "hun",
      "score": 3.706084498844575e-06
    },
    {
      "language": "Haitian Creole",
      "code": "hat",
      "score": 3.6860656109638512e-06
    },
    {
      "language": "Jamaican Patois",
      "code": "jam",
      "score": 3.6750652725459076e-06
    },
    {
      "language": "Turkmen",
      "code": "tuk",
      "score": 3.6414037367649144e-06
    },
    {
      "language": "Gagauz",
      "code": "gag",
      "score": 3.6310443647380453e-06
    },
    {
      "language": "Yakut",
      "code": "sah",
      "score": 3.611620968513307e-06
    },
    {
      "language": "Breton",
      "code": "bre",
      "score": 3.5204120649723336e-06
    },
    {
      "language": "Afrikaans",
      "code": "afr",
      "score": 3.5164177916158224e-06
    },
    {
      "language": "Assamese",
      "code": "asm",
      "score": 3.5076063795713708e-06
    },
    {
      "language": "Crimean Tatar",
      "code": "crh",
      "score": 3.4974791560671292e-06
    },
    {
      "language": "Tswana",
      "code": "tsn",
      "score": 3.4639840578165604e-06
    },
    {
      "language": "Malagasy",
      "code": "mlg",
      "score": 3.4424308523739455e-06
    },
    {
      "language": "Tamil",
      "code": "tam",
      "score": 3.433554866205668e-06
    },
    {
      "language": "Belarusian (Taraschkewiza)",
      "code": "be-tarask",
      "score": 3.4065565159835387e-06
    },
    {
      "language": "Scottish Gaelic",
      "code": "gla",
      "score": 3.383374632903724e-06
    },
    {
      "language": "Latin",
      "code": "lat",
      "score": 3.299320724181598e-06
    },
    {
      "language": "Chavacano",
      "code": "cbk",
      "score": 3.277132236689795e-06
    },
    {
      "language": "Tarantino dialect",
      "code": "roa-tara",
      "score": 3.2704483601264656e-06
    },
    {
      "language": "Modern Greek",
      "code": "ell",
      "score": 3.2669522624928504e-06
    },
    {
      "language": "Ladino",
      "code": "lad",
      "score": 3.1890219815977616e-06
    },
    {
      "language": "Latgalian",
      "code": "ltg",
      "score": 3.1830948046263075e-06
    },
    {
      "language": "Pampanga",
      "code": "pam",
      "score": 3.1460281206818763e-06
    },
    {
      "language": "Tagalog",
      "code": "tgl",
      "score": 3.100457433902193e-06
    },
    {
      "language": "Hebrew",
      "code": "heb",
      "score": 3.0715009415871464e-06
    },
    {
      "language": "Serbo-Croatian",
      "code": "hbs",
      "score": 3.050950908800587e-06
    },
    {
      "language": "Achinese",
      "code": "ace",
      "score": 3.0138855890982086e-06
    },
    {
      "language": "Italian",
      "code": "ita",
      "score": 3.003329993589432e-06
    },
    {
      "language": "English",
      "code": "eng",
      "score": 2.97778979074792e-06
    },
    {
      "language": "Burmese",
      "code": "mya",
      "score": 2.9546490623033606e-06
    },
    {
      "language": "Spanish",
      "code": "spa",
      "score": 2.9272057417983888e-06
    },
    {
      "language": "Papiamento",
      "code": "pap",
      "score": 2.8780641514458694e-06
    },
    {
      "language": "Sardinian",
      "code": "srd",
      "score": 2.866505383281037e-06
    },
    {
      "language": "Esperanto",
      "code": "epo",
      "score": 2.848199301297427e-06
    },
    {
      "language": "Serbian",
      "code": "srp",
      "score": 2.7479175059852423e-06
    },
    {
      "language": "Zeeuws",
      "code": "zea",
      "score": 2.7430314730736427e-06
    },
    {
      "language": "Czech",
      "code": "ces",
      "score": 2.7409500944486354e-06
    },
    {
      "language": "Bengali",
      "code": "ben",
      "score": 2.6958239232044434e-06
    },
    {
      "language": "Erzya",
      "code": "myv",
      "score": 2.6273187359038275e-06
    },
    {
      "language": "Croatian",
      "code": "hrv",
      "score": 2.6178654479735997e-06
    },
    {
      "language": "Buryat",
      "code": "bxr",
      "score": 2.60430465459649e-06
    },
    {
      "language": "Swahili (macrolanguage)",
      "code": "swa",
      "score": 2.6016373340098653e-06
    },
    {
      "language": "Pangasinan",
      "code": "pag",
      "score": 2.60037768384791e-06
    },
    {
      "language": "Xhosa",
      "code": "xho",
      "score": 2.580123918960453e-06
    },
    {
      "language": "Bosnian",
      "code": "bos",
      "score": 2.5763115445442963e-06
    },
    {
      "language": "Low German",
      "code": "nds",
      "score": 2.5743340756889665e-06
    },
    {
      "language": "Kinyarwanda",
      "code": "kin",
      "score": 2.568235458966228e-06
    },
    {
      "language": "Aromanian",
      "code": "rup",
      "score": 2.520287125662435e-06
    },
    {
      "language": "Aragonese",
      "code": "arg",
      "score": 2.4836215288814856e-06
    },
    {
      "language": "Tetum",
      "code": "tet",
      "score": 2.396502168267034e-06
    },
    {
      "language": "Quechua",
      "code": "que",
      "score": 2.3799134396540467e-06
    },
    {
      "language": "Livvi-Karelian",
      "code": "olo",
      "score": 2.3709426386631094e-06
    },
    {
      "language": "Kashubian",
      "code": "csb",
      "score": 2.358733354412834e-06
    },
    {
      "language": "Avar",
      "code": "ava",
      "score": 2.330698407604359e-06
    },
    {
      "language": "Hausa",
      "code": "hau",
      "score": 2.286114295202424e-06
    },
    {
      "language": "Ripuarisch",
      "code": "ksh",
      "score": 2.254129412904149e-06
    },
    {
      "language": "Bulgarian",
      "code": "bul",
      "score": 2.2492179141408997e-06
    },
    {
      "language": "Oriya",
      "code": "ori",
      "score": 2.1661755909008207e-06
    },
    {
      "language": "Interlingue",
      "code": "ile",
      "score": 2.059975486190524e-06
    },
    {
      "language": "Guarani",
      "code": "grn",
      "score": 2.024690957114217e-06
    },
    {
      "language": "Banjar",
      "code": "bjn",
      "score": 2.0237362150510307e-06
    },
    {
      "language": "Thai",
      "code": "tha",
      "score": 2.01868806470884e-06
    },
    {
      "language": "Dutch",
      "code": "nld",
      "score": 1.9297158360132016e-06
    },
    {
      "language": "Kabyle",
      "code": "kab",
      "score": 1.9132662600895856e-06
    },
    {
      "language": "Palatine German",
      "code": "pfl",
      "score": 1.9122355752188014e-06
    },
    {
      "language": "Javanese",
      "code": "jav",
      "score": 1.8900879013017402e-06
    },
    {
      "language": "Banyumasan",
      "code": "map-bms",
      "score": 1.8552185565567925e-06
    },
    {
      "language": "Faroese",
      "code": "fao",
      "score": 1.8414674514133367e-06
    },
    {
      "language": "Scots",
      "code": "sco",
      "score": 1.818199393710529e-06
    },
    {
      "language": "Central Khmer",
      "code": "khm",
      "score": 1.7993022538576042e-06
    },
    {
      "language": "Slovak",
      "code": "slk",
      "score": 1.7988603531193803e-06
    },
    {
      "language": "Belarusian",
      "code": "bel",
      "score": 1.782583581189101e-06
    },
    {
      "language": "Swedish",
      "code": "swe",
      "score": 1.7702136574371252e-06
    },
    {
      "language": "Saterfriesisch",
      "code": "stq",
      "score": 1.7663436437942437e-06
    },
    {
      "language": "Latvian",
      "code": "lav",
      "score": 1.7178032294395962e-06
    },
    {
      "language": "Konkani",
      "code": "kok",
      "score": 1.690383783170546e-06
    },
    {
      "language": "Tuvan",
      "code": "tyv",
      "score": 1.672853159107035e-06
    },
    {
      "language": "Walloon",
      "code": "wln",
      "score": 1.6722132158975e-06
    },
    {
      "language": "Sranan",
      "code": "srn",
      "score": 1.646132773203135e-06
    },
    {
      "language": "Picard",
      "code": "pcd",
      "score": 1.6385885146519286e-06
    },
    {
      "language": "Limburgan",
      "code": "lim",
      "score": 1.6372666777897393e-06
    },
    {
      "language": "French",
      "code": "fra",
      "score": 1.6239549722740776e-06
    },
    {
      "language": "Icelandic",
      "code": "isl",
      "score": 1.5904075780781568e-06
    },
    {
      "language": "Irish",
      "code": "gle",
      "score": 1.5750525790281245e-06
    },
    {
      "language": "Corsican",
      "code": "cos",
      "score": 1.570832523611898e-06
    },
    {
      "language": "Alemannic German",
      "code": "als",
      "score": 1.5651218063794659e-06
    },
    {
      "language": "German",
      "code": "deu",
      "score": 1.5594737305946182e-06
    },
    {
      "language": "Upper Sorbian",
      "code": "hsb",
      "score": 1.5125158370210556e-06
    },
    {
      "language": "Romanian",
      "code": "ron",
      "score": 1.5119784393391456e-06
    },
    {
      "language": "Manx",
      "code": "glv",
      "score": 1.5035052456369158e-06
    },
    {
      "language": "Lingala",
      "code": "lin",
      "score": 1.493238073635439e-06
    },
    {
      "language": "Malay",
      "code": "msa",
      "score": 1.4067626352698426e-06
    },
    {
      "language": "Maltese",
      "code": "mlt",
      "score": 1.370485165352875e-06
    },
    {
      "language": "Luxembourgish",
      "code": "ltz",
      "score": 1.3397349221122568e-06
    },
    {
      "language": "Estonian",
      "code": "est",
      "score": 1.3280839539220324e-06
    },
    {
      "language": "Kabardian",
      "code": "kbd",
      "score": 1.3062604011793155e-06
    },
    {
      "language": "Macedonian",
      "code": "mkd",
      "score": 1.2802570381609257e-06
    },
    {
      "language": "Pennsylvania German",
      "code": "pdc",
      "score": 1.2550040082714986e-06
    },
    {
      "language": "Sundanese",
      "code": "sun",
      "score": 1.1068191270169336e-06
    },
    {
      "language": "Iloko",
      "code": "ilo",
      "score": 1.0791690101541462e-06
    },
    {
      "language": "Karakalpak",
      "code": "kaa",
      "score": 1.0603262126096524e-06
    },
    {
      "language": "Norwegian Nynorsk",
      "code": "nno",
      "score": 1.0554679192864569e-06
    },
    {
      "language": "Yoruba",
      "code": "yor",
      "score": 1.046297711582156e-06
    },
    {
      "language": "Neapolitan",
      "code": "nap",
      "score": 1.0279602520313347e-06
    },
    {
      "language": "Danish",
      "code": "dan",
      "score": 1.0038916116172913e-06
    },
    {
      "language": "Indonesian",
      "code": "ind",
      "score": 9.83746303973021e-07
    },
    {
      "language": "Mirandese",
      "code": "mwl",
      "score": 8.806521236692788e-07
    },
    {
      "language": "Catalan",
      "code": "cat",
      "score": 8.687447348165733e-07
    },
    {
      "language": "Turkish",
      "code": "tur",
      "score": 8.384120064874878e-07
    },
    {
      "language": "Veps",
      "code": "vep",
      "score": 7.812500371073838e-07
    },
    {
      "language": "Bokm\u00e5l",
      "code": "nob",
      "score": 7.427178161378833e-07
    },
    {
      "language": "Shona",
      "code": "sna",
      "score": 6.660703775196453e-07
    },
    {
      "language": "Bavarian",
      "code": "bar",
      "score": 6.222485353646334e-07
    },
    {
      "language": "Uzbek",
      "code": "uzb",
      "score": 6.021850822435226e-07
    },
    {
      "language": "Central Bikol",
      "code": "bcl",
      "score": 5.77034370508045e-07
    },
    {
      "language": "Asturian",
      "code": "ast",
      "score": 5.743918336520437e-07
    },
    {
      "language": "Lombard",
      "code": "lmo",
      "score": 4.6301857992148143e-07
    },
    {
      "language": "Romansh",
      "code": "roh",
      "score": 4.5534079617937095e-07
    },
    {
      "language": "Narom",
      "code": "nrm",
      "score": 3.6611126574825903e-07
    },
    {
      "language": "Northern Sami",
      "code": "sme",
      "score": 1.0723972820869676e-07
    }
  ]
]
```

</details>

## Citation

Please cite this repository in publications as the following:

```bibtex
@misc{ZabanShenas,
  author       = {Mehrdad Farahani},
  title        = {Zabanshenas is a solution for identifying the most likely language of a piece of written text},
  month        = feb,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v2.0.1},
  doi          = {10.5281/zenodo.5029022},
  url          = {https://doi.org/10.5281/zenodo.5029022}
}
```


## License
[Apache License 2.0](LICENSE)
