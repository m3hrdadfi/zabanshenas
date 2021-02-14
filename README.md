<h1 align="center">Zabanshenas</h1>

zabanshenas (زبان‌شناس / zæbænʒenæs) which has two meanings

- A person who studies linguistics.
- A way to identify the type of written language.

<br/><br/>

**Table of Contents**
- [Introduction](#introduction)
- [WilI-2018 (Cleaned version)](#wili-2018-cleaned-version)
- [Model (Architecture)](#model-architecture)
- [Training Configuration](#training-configuration)
- [Evaluation](#evaluation)
- [How to Install](#how-to-install)
  - [Install with pip](#install-with-pip)
  - [Install from sources](#install-from-sources)
- [Getting Started](#getting-started)

# Introduction 
This repository is a preliminary step in creating a language detection script that inherits benefits from [Transformer](https://arxiv.org/abs/1706.03762) architecture. Nowadays, Transformers have played a massive role in Natural Language Processing fields. Simply, Transformers uses an attention mechanism to boost the speed and extract a high level of information (abstraction).

There are plenty of ways, solutions, and packages to find the language of a written piece of text or document. All of them have their pros and cons. Some able to detect faster and support as many languages as possible. However, in this case, I plan to implement a scenario capable of understanding similar groups of languages and covering more than 200 languages (235 exactly) thanks to [WiLI-2018](https://arxiv.org/abs/1801.07779) and the Transformer architecture.


# WilI-2018 (Cleaned version)
It is a benchmark dataset for monolingual written natural language identification (high covering of a specific language). It contains 229,095 paragraphs that cover 235 languages. Language distribution includes 122 Indo-European languages, 22 Austronesian languages, 17 Turkic languages, 14 Uralic languages, 11 Niger-Congo languages, 10 Sino-Tibetan languages, 9 Afro-Asiatic languages, 6 constructed languages, and 24 languages of smaller families. It also consists of groups of similar languages and dialects:
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


# Model (Architecture)

<p align="center">
    <img src="reports/transformer.png" alt="">
    <br>
    <em>Fig 5: The Transformer - model architecture.</em>
</p>

Transformers are a kind of neural sequence transduction model that has a couple of encoders and decoders. The encoders map inputs as sequential representations to continuous representations of Z and then fed Z to decoders in order to generate another sequential representation as outputs, as shown in the figure above. What makes this neural sequence transduction so special is a new attention mechanism (Multi-Head Attention) that allows the model to extract high-level information, as shown in Fig 6. In this particular case (Language Detection Model), the combination of Transformer-Encoders with a mean-pooling softmax-classifier as a detector could be very effective.

<p align="center">
    <img src="reports/mha.png" alt="">
    <br>
    <em>Fig 6: Scaled Dot-Product Attention / Multi-Head Attention.</em>
</p>


# Training Configuration
In the training step, the model setup for the following configurations:
- d_model: 512
- heads: 8
- d_ff: 1024
- n_position: 5000
- n_layers: 3
- pooling_strategy: mean
- n_classes: 235
- ff_activation: ReLU
- ff_bias1: True
- ff_bias2: True
- wte_normalization: True
- max_seq_len: 512
- enable_padding: True
- learning_rate: 1e-4
- n_epochs: 5

# Evaluation
The model evaluated over 22,910 samples and tested over 45,819 that both samplings covered 235 languages and precision, recall, F1-scores for each language presented in the following table.

|                language                | precision | recall | f1-score |
|:--------------------------------------:|:---------:|:------:|:--------:|
|             Achinese (ace)             |    0.99   |  0.91  |   0.95   |
|             Afrikaans (afr)            |    0.91   |  0.96  |   0.93   |
|         Alemannic German (als)         |    0.89   |  0.74  |   0.81   |
|              Amharic (amh)             |    1.00   |  0.97  |   0.98   |
|           Old English  (ang)           |    0.88   |  0.85  |   0.86   |
|              Arabic (ara)              |    0.91   |  0.93  |   0.92   |
|             Aragonese (arg)            |    0.83   |  0.90  |   0.86   |
|          Egyptian Arabic (arz)         |    0.93   |  0.91  |   0.92   |
|             Assamese (asm)             |    0.99   |  0.98  |   0.98   |
|             Asturian (ast)             |    0.92   |  0.67  |   0.78   |
|               Avar (ava)               |    0.65   |  0.73  |   0.69   |
|              Aymara (aym)              |    0.94   |  0.88  |   0.91   |
|         South Azerbaijani (azb)        |    0.99   |  0.95  |   0.97   |
|            Azerbaijani (aze)           |    1.00   |  0.94  |   0.97   |
|              Bashkir (bak)             |    0.99   |  0.91  |   0.95   |
|             Bavarian (bar)             |    0.81   |  0.80  |   0.80   |
|           Central Bikol (bcl)          |    0.98   |  0.92  |   0.95   |
| Belarusian (Taraschkewiza) (be-tarask) |    0.67   |  0.75  |   0.71   |
|            Belarusian (bel)            |    0.71   |  0.56  |   0.63   |
|              Bengali (ben)             |    0.88   |  0.90  |   0.89   |
|             Bhojpuri (bho)             |    0.78   |  0.87  |   0.82   |
|              Banjar (bjn)              |    0.93   |  0.66  |   0.77   |
|              Tibetan (bod)             |    1.00   |  0.99  |   0.99   |
|              Bosnian (bos)             |    0.36   |  0.76  |   0.49   |
|            Bishnupriya (bpy)           |    0.97   |  0.97  |   0.97   |
|              Breton (bre)              |    0.97   |  0.95  |   0.96   |
|             Bulgarian (bul)            |    0.91   |  0.92  |   0.91   |
|              Buryat (bxr)              |    0.92   |  0.79  |   0.85   |
|              Catalan (cat)             |    0.76   |  0.92  |   0.84   |
|             Chavacano (cbk)            |    0.69   |  0.40  |   0.51   |
|             Min Dong (cdo)             |    0.89   |  0.97  |   0.93   |
|              Cebuano (ceb)             |    0.95   |  1.00  |   0.97   |
|               Czech (ces)              |    0.96   |  0.92  |   0.94   |
|              Chechen (che)             |    0.77   |  0.80  |   0.78   |
|             Cherokee (chr)             |    1.00   |  0.95  |   0.98   |
|              Chuvash (chv)             |    0.98   |  0.91  |   0.95   |
|          Central Kurdish (ckb)         |    0.98   |  0.98  |   0.98   |
|              Cornish (cor)             |    0.96   |  0.95  |   0.95   |
|             Corsican (cos)             |    0.90   |  0.94  |   0.92   |
|           Crimean Tatar (crh)          |    0.86   |  0.93  |   0.89   |
|             Kashubian (csb)            |    0.93   |  0.97  |   0.95   |
|               Welsh (cym)              |    0.96   |  0.95  |   0.96   |
|              Danish (dan)              |    0.93   |  0.83  |   0.87   |
|              German (deu)              |    0.56   |  0.91  |   0.69   |
|               Dimli (diq)              |    0.98   |  0.89  |   0.93   |
|              Dhivehi (div)             |    1.00   |  0.99  |   0.99   |
|           Lower Sorbian (dsb)          |    0.86   |  0.86  |   0.86   |
|              Doteli (dty)              |    0.55   |  0.67  |   0.60   |
|              Emilian (egl)             |    0.82   |  0.89  |   0.85   |
|           Modern Greek (ell)           |    0.93   |  0.98  |   0.96   |
|              English (eng)             |    0.26   |  0.94  |   0.40   |
|             Esperanto (epo)            |    0.75   |  0.94  |   0.83   |
|             Estonian (est)             |    0.65   |  0.86  |   0.74   |
|              Basque (eus)              |    0.91   |  0.96  |   0.93   |
|           Extremaduran (ext)           |    0.92   |  0.72  |   0.80   |
|              Faroese (fao)             |    0.92   |  0.95  |   0.94   |
|              Persian (fas)             |    0.81   |  0.92  |   0.86   |
|              Finnish (fin)             |    0.84   |  0.88  |   0.86   |
|              French (fra)              |    0.52   |  0.89  |   0.65   |
|              Arpitan (frp)             |    0.77   |  0.78  |   0.77   |
|          Western Frisian (fry)         |    0.83   |  0.95  |   0.89   |
|             Friulian (fur)             |    0.93   |  0.76  |   0.83   |
|              Gagauz (gag)              |    0.91   |  0.79  |   0.85   |
|          Scottish Gaelic (gla)         |    0.86   |  0.96  |   0.91   |
|               Irish (gle)              |    0.91   |  0.94  |   0.92   |
|             Galician (glg)             |    0.85   |  0.79  |   0.82   |
|              Gilaki (glk)              |    0.86   |  0.87  |   0.86   |
|               Manx (glv)               |    0.98   |  0.93  |   0.96   |
|              Guarani (grn)             |    0.99   |  0.95  |   0.97   |
|             Gujarati (guj)             |    1.00   |  0.93  |   0.97   |
|           Hakka Chinese (hak)          |    0.99   |  0.95  |   0.97   |
|          Haitian Creole (hat)          |    0.60   |  0.89  |   0.72   |
|               Hausa (hau)              |    0.95   |  0.93  |   0.94   |
|          Serbo-Croatian (hbs)          |    0.41   |  0.22  |   0.29   |
|              Hebrew (heb)              |    0.98   |  0.99  |   0.98   |
|            Fiji Hindi (hif)            |    0.88   |  0.89  |   0.89   |
|               Hindi (hin)              |    0.88   |  0.83  |   0.85   |
|             Croatian (hrv)             |    0.41   |  0.13  |   0.20   |
|           Upper Sorbian (hsb)          |    0.81   |  0.89  |   0.85   |
|             Hungarian (hun)            |    0.86   |  0.99  |   0.92   |
|             Armenian (hye)             |    0.99   |  0.95  |   0.97   |
|               Igbo (ibo)               |    0.94   |  0.80  |   0.86   |
|                Ido (ido)               |    0.92   |  0.91  |   0.91   |
|            Interlingue (ile)           |    0.98   |  0.85  |   0.91   |
|               Iloko (ilo)              |    0.90   |  0.90  |   0.90   |
|            Interlingua (ina)           |    0.93   |  0.95  |   0.94   |
|            Indonesian (ind)            |    0.33   |  0.01  |   0.01   |
|             Icelandic (isl)            |    0.97   |  0.95  |   0.96   |
|              Italian (ita)             |    0.80   |  0.86  |   0.83   |
|          Jamaican Patois (jam)         |    0.83   |  0.97  |   0.89   |
|             Javanese (jav)             |    0.85   |  0.85  |   0.85   |
|              Lojban (jbo)              |    1.00   |  0.96  |   0.98   |
|             Japanese (jpn)             |    0.99   |  0.97  |   0.98   |
|            Karakalpak (kaa)            |    0.99   |  0.90  |   0.94   |
|              Kabyle (kab)              |    0.93   |  0.88  |   0.90   |
|              Kannada (kan)             |    0.90   |  0.96  |   0.93   |
|             Georgian (kat)             |    0.97   |  0.94  |   0.95   |
|              Kazakh (kaz)              |    0.96   |  0.96  |   0.96   |
|             Kabardian (kbd)            |    0.98   |  0.98  |   0.98   |
|           Central Khmer (khm)          |    0.95   |  0.89  |   0.92   |
|            Kinyarwanda (kin)           |    0.93   |  0.86  |   0.89   |
|              Kirghiz (kir)             |    0.90   |  0.89  |   0.90   |
|           Komi-Permyak (koi)           |    0.85   |  0.87  |   0.86   |
|              Konkani (kok)             |    0.80   |  0.77  |   0.79   |
|               Komi (kom)               |    0.91   |  0.79  |   0.85   |
|              Korean (kor)              |    0.99   |  0.98  |   0.99   |
|          Karachay-Balkar (krc)         |    0.91   |  0.91  |   0.91   |
|            Ripuarisch (ksh)            |    0.67   |  0.88  |   0.76   |
|              Kurdish (kur)             |    0.90   |  0.98  |   0.94   |
|              Ladino (lad)              |    0.86   |  0.89  |   0.87   |
|                Lao (lao)               |    0.98   |  0.88  |   0.93   |
|               Latin (lat)              |    0.90   |  0.86  |   0.88   |
|              Latvian (lav)             |    0.94   |  0.88  |   0.91   |
|             Lezghian (lez)             |    0.96   |  0.87  |   0.91   |
|             Ligurian (lij)             |    0.92   |  0.89  |   0.90   |
|             Limburgan (lim)            |    0.76   |  0.88  |   0.81   |
|              Lingala (lin)             |    0.89   |  0.91  |   0.90   |
|            Lithuanian (lit)            |    0.95   |  0.94  |   0.94   |
|              Lombard (lmo)             |    0.92   |  0.77  |   0.84   |
|           Northern Luri (lrc)          |    0.94   |  0.75  |   0.84   |
|             Latgalian (ltg)            |    0.88   |  0.93  |   0.90   |
|           Luxembourgish (ltz)          |    0.84   |  0.81  |   0.82   |
|              Luganda (lug)             |    0.92   |  0.90  |   0.91   |
|         Literary Chinese (lzh)         |    0.91   |  0.94  |   0.93   |
|             Maithili (mai)             |    0.84   |  0.92  |   0.88   |
|             Malayalam (mal)            |    1.00   |  0.97  |   0.98   |
|          Banyumasan (map-bms)          |    0.83   |  0.77  |   0.80   |
|              Marathi (mar)             |    0.95   |  0.81  |   0.87   |
|              Moksha (mdf)              |    0.85   |  0.95  |   0.90   |
|           Eastern Mari (mhr)           |    0.90   |  0.85  |   0.88   |
|            Minangkabau (min)           |    0.98   |  0.95  |   0.97   |
|            Macedonian (mkd)            |    0.95   |  0.91  |   0.93   |
|             Malagasy (mlg)             |    0.99   |  0.97  |   0.98   |
|              Maltese (mlt)             |    0.91   |  0.98  |   0.94   |
|             Mongolian (mon)            |    0.89   |  0.96  |   0.92   |
|               Maori (mri)              |    0.99   |  0.97  |   0.98   |
|           Western Mari (mrj)           |    0.99   |  0.97  |   0.98   |
|               Malay (msa)              |    0.39   |  0.96  |   0.56   |
|             Mirandese (mwl)            |    0.97   |  0.93  |   0.95   |
|              Burmese (mya)             |    1.00   |  1.00  |   1.00   |
|               Erzya (myv)              |    0.66   |  0.73  |   0.69   |
|            Mazanderani (mzn)           |    0.86   |  0.91  |   0.89   |
|          Min Nan Chinese (nan)         |    0.97   |  0.95  |   0.96   |
|            Neapolitan (nap)            |    0.97   |  0.70  |   0.81   |
|              Navajo (nav)              |    0.99   |  0.99  |   0.99   |
|         Classical Nahuatl (nci)        |    0.98   |  0.54  |   0.70   |
|            Low German (nds)            |    0.87   |  0.91  |   0.89   |
|        West Low German (nds-nl)        |    0.68   |  0.80  |   0.73   |
|      Nepali (macrolanguage) (nep)      |    0.68   |  0.51  |   0.58   |
|              Newari (new)              |    0.93   |  0.79  |   0.86   |
|               Dutch (nld)              |    0.86   |  0.80  |   0.83   |
|         Norwegian Nynorsk (nno)        |    0.84   |  0.83  |   0.83   |
|              Bokmål (nob)              |    0.71   |  0.83  |   0.76   |
|               Narom (nrm)              |    0.95   |  0.70  |   0.80   |
|          Northern Sotho (nso)          |    0.79   |  0.96  |   0.87   |
|              Occitan (oci)             |    0.82   |  0.57  |   0.67   |
|          Livvi-Karelian (olo)          |    0.85   |  0.91  |   0.88   |
|               Oriya (ori)              |    0.99   |  0.95  |   0.97   |
|               Oromo (orm)              |    0.96   |  0.86  |   0.90   |
|             Ossetian (oss)             |    0.99   |  0.94  |   0.97   |
|            Pangasinan (pag)            |    0.98   |  0.82  |   0.89   |
|             Pampanga (pam)             |    0.96   |  0.63  |   0.76   |
|              Panjabi (pan)             |    1.00   |  0.99  |   0.99   |
|            Papiamento (pap)            |    0.91   |  0.94  |   0.92   |
|              Picard (pcd)              |    0.72   |  0.58  |   0.64   |
|        Pennsylvania German (pdc)       |    0.74   |  0.69  |   0.72   |
|          Palatine German (pfl)         |    0.80   |  0.67  |   0.73   |
|          Western Panjabi (pnb)         |    0.94   |  0.97  |   0.96   |
|              Polish (pol)              |    0.92   |  0.92  |   0.92   |
|            Portuguese (por)            |    0.75   |  0.84  |   0.79   |
|              Pushto (pus)              |    0.99   |  0.94  |   0.96   |
|              Quechua (que)             |    0.74   |  0.93  |   0.82   |
|      Tarantino dialect (roa-tara)      |    0.92   |  0.90  |   0.91   |
|              Romansh (roh)             |    0.90   |  0.93  |   0.91   |
|             Romanian (ron)             |    0.97   |  0.89  |   0.93   |
|               Rusyn (rue)              |    0.95   |  0.84  |   0.90   |
|             Aromanian (rup)            |    0.95   |  0.82  |   0.88   |
|              Russian (rus)             |    0.64   |  0.81  |   0.72   |
|               Yakut (sah)              |    0.97   |  0.93  |   0.95   |
|             Sanskrit (san)             |    0.98   |  0.90  |   0.94   |
|             Sicilian (scn)             |    0.85   |  0.87  |   0.86   |
|               Scots (sco)              |    0.95   |  0.88  |   0.91   |
|            Samogitian (sgs)            |    0.98   |  0.96  |   0.97   |
|              Sinhala (sin)             |    0.99   |  0.91  |   0.95   |
|              Slovak (slk)              |    0.95   |  0.89  |   0.92   |
|              Slovene (slv)             |    0.95   |  0.87  |   0.91   |
|           Northern Sami (sme)          |    0.88   |  0.90  |   0.89   |
|               Shona (sna)              |    0.88   |  0.82  |   0.85   |
|              Sindhi (snd)              |    0.98   |  0.98  |   0.98   |
|              Somali (som)              |    0.90   |  0.84  |   0.87   |
|              Spanish (spa)             |    0.46   |  0.84  |   0.59   |
|             Albanian (sqi)             |    0.95   |  0.93  |   0.94   |
|             Sardinian (srd)            |    0.94   |  0.92  |   0.93   |
|              Sranan (srn)              |    0.98   |  0.88  |   0.93   |
|              Serbian (srp)             |    0.91   |  0.87  |   0.88   |
|          Saterfriesisch (stq)          |    0.96   |  0.91  |   0.93   |
|             Sundanese (sun)            |    0.93   |  0.86  |   0.89   |
|      Swahili (macrolanguage) (swa)     |    0.93   |  0.85  |   0.88   |
|              Swedish (swe)             |    0.94   |  0.92  |   0.93   |
|             Silesian (szl)             |    0.92   |  0.97  |   0.95   |
|               Tamil (tam)              |    0.99   |  0.98  |   0.98   |
|               Tatar (tat)              |    0.95   |  0.91  |   0.93   |
|               Tulu (tcy)               |    0.95   |  0.91  |   0.93   |
|              Telugu (tel)              |    0.99   |  0.94  |   0.97   |
|               Tetum (tet)              |    1.00   |  0.81  |   0.90   |
|               Tajik (tgk)              |    0.96   |  0.94  |   0.95   |
|              Tagalog (tgl)             |    0.93   |  0.93  |   0.93   |
|               Thai (tha)               |    0.90   |  0.97  |   0.93   |
|              Tongan (ton)              |    0.99   |  0.99  |   0.99   |
|              Tswana (tsn)              |    0.99   |  0.69  |   0.82   |
|              Turkmen (tuk)             |    0.98   |  0.94  |   0.96   |
|              Turkish (tur)             |    0.84   |  0.86  |   0.85   |
|               Tuvan (tyv)              |    0.89   |  0.90  |   0.90   |
|              Udmurt (udm)              |    0.98   |  0.94  |   0.96   |
|              Uighur (uig)              |    0.98   |  0.96  |   0.97   |
|             Ukrainian (ukr)            |    0.81   |  0.94  |   0.87   |
|               Urdu (urd)               |    0.98   |  0.97  |   0.98   |
|               Uzbek (uzb)              |    0.97   |  0.92  |   0.94   |
|             Venetian (vec)             |    0.88   |  0.86  |   0.87   |
|               Veps (vep)               |    0.94   |  0.90  |   0.92   |
|            Vietnamese (vie)            |    0.97   |  0.94  |   0.95   |
|              Vlaams (vls)              |    0.92   |  0.65  |   0.76   |
|              Volapük (vol)             |    0.98   |  0.99  |   0.98   |
|               Võro (vro)               |    0.98   |  0.76  |   0.85   |
|               Waray (war)              |    1.00   |  0.98  |   0.99   |
|              Walloon (wln)             |    0.96   |  0.91  |   0.93   |
|               Wolof (wol)              |    0.94   |  0.90  |   0.92   |
|            Wu Chinese (wuu)            |    0.95   |  0.85  |   0.90   |
|               Xhosa (xho)              |    0.91   |  0.85  |   0.88   |
|            Mingrelian (xmf)            |    0.95   |  0.96  |   0.95   |
|              Yiddish (yid)             |    0.99   |  0.96  |   0.98   |
|              Yoruba (yor)              |    0.87   |  0.84  |   0.85   |
|              Zeeuws (zea)              |    0.78   |  0.75  |   0.76   |
|           Cantonese (zh-yue)           |    0.87   |  0.90  |   0.89   |
|         Standard Chinese (zho)         |    0.92   |  0.92  |   0.92   |
|                  MEAN                  |    0.89   |  0.87  |   0.87   |


As can be seen, the model outperforms on groups of similar or dialects languages. For instance, the f1 score for the Persian language with other dialects Gilaki and Mazanderani is 86%, 86%, and 89%, respectively. Another example in other similar languages like Arabic and Egyptian Arabic 92% and 92% shows the model achieved tremendous results.

To better analyze the model's result, it is necessary to refer to the confusion matrix shown in the figure below Fig 7.


<p align="center">
    <img src="reports/cm.png" alt="">
    <br>
    <em>Fig 7: Confusion matrix of Transformer-Mean-Pooling Classifier. The diagonal was set to 0 to emphasize the errors. The y-axis is the actual language, the x-axis the predicted language.</em>
</p>


# How to Install
We recommend Python 3.7 or higher, PyTorch 1.6.0 or higher. The code does not work with Python 2.7.

## Install with pip
So...on

## Install from sources
You can also clone the latest version from the repository and install it directly from the source code:

```bash
git clone https://github.com/m3hrdadfi/zabanshenas
cd zabanshenas
pip install -e .
```

# Getting Started

You can use this code snippet to identify the most likely language of a written document. You just have to say: ZABANSHENAS (language detector) -> BESHNAS (detect) 😎. 

*Sounds interesting, doesn't it?*

```python
from zabanshenas import beshnas

text = "خورشت فسنجان یا خورشت فسنجون یکی از خورش‌های اصیل ایرانی است که از غذاهای سنتی و باستانی گیلان محسوب می‌شود که در همه جای ایران باب شده است."
beshnas(text)

>>> {'code': 'fas', 'idx': 56, 'name': 'Persian', 'prob': 0.9846975803375244}

text = "Pakistan isch no dr Verfassuig vo 1973 e föderale Staat. Är untergliederet sich in di vier Provinze Belutschistan, Khyber Pakhtunkhwa (früener Nordweschtlichi Gränzprovinz), Punjab un Sindh, wo ne uf "

beshnas(text)

>>> {'code': 'als', 'idx': 2, 'name': 'Alemannic German', 'prob': 0.7985869646072388}

text = "Dario Fo hett 1954 de Schauspelerin un latere politische Aktivistin Franca Rame (1929–2013) heiraadt, mit de he ok künstlerisch eng tosommenarbeit hett."

beshnas(text)

>>> {'code': 'nds', 'idx': 146, 'name': 'Low German', 'prob': 0.6278020739555359}
```

Or you can find out the k-possible candidates of detected languages using the following snippet.

```python
from zabanshenas import beshnas

text = "خورشت فسنجان یا خورشت فسنجون یکی از خورش‌های اصیل ایرانی است که از غذاهای سنتی و باستانی گیلان محسوب می‌شود که در همه جای ایران باب شده است."
beshnas(text, show_topk=True)

>>> 
{0: {'code': 'fas', 'idx': 56, 'name': 'Persian', 'prob': 0.9846975803375244},
 1: {'code': 'mzn',
     'idx': 141,
     'name': 'Mazanderani',
     'prob': 0.011471129022538662},
 2: {'code': 'lrc',
     'idx': 119,
     'name': 'Northern Luri',
     'prob': 0.0017076199874281883},
 3: {'code': 'glk', 'idx': 66, 'name': 'Gilaki', 'prob': 0.001486123655922711},
 4: {'code': 'ckb',
     'idx': 36,
     'name': 'Central Kurdish',
     'prob': 0.00019319159036967903}}
```