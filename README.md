# Portfolio Optimization: Comparing Covariance Denoising Methods to Basic Method

## Folder Structure

```
.
├─ .DS_Store
├─ README.md
├─ bahc_method.ipynb
├─ data
│  ├─ .DS_Store
│  ├─ bahc
│  │  ├─ .DS_Store
│  │  ├─ moving_avg_in_sample.csv
│  │  ├─ moving_avg_out_sample.csv
│  │  ├─ risk_out_sample.csv
│  │  ├─ risks_in_sample.csv
│  │  └─ weights.csv
│  ├─ clean_full_bbo_data.parquet
│  ├─ correlation_data
│  │  ├─ .DS_Store
│  │  ├─ moving_avg_in_sample.csv
│  │  ├─ moving_avg_out_sample.csv
│  │  ├─ risk_out_sample.csv
│  │  ├─ risks_in_sample.csv
│  │  └─ weights.csv
│  └─ covariance_data
│     ├─ .DS_Store
│     ├─ moving_avg_in_sample.csv
│     ├─ moving_avg_out_sample.csv
│     ├─ risks_in_sample.csv
│     ├─ risks_out_sample.csv
│     └─ weights_df.csv
├─ data_analysis.ipynb
├─ get_smaller_dataset.ipynb
├─ load_data.py
├─ main.ipynb
├─ requirements.txt
└─ utils.py
```




## [Project Setup](#setup)
Install required packages with :
`pip install -r requirements.txt`

Set up a data folder as following : 
- `mkdir -p data`
  
Subsequently, download datafrom the [drive](https://drive.switch.ch/index.php/s/0X3Je6DauQRzD2r) provided by the Professor. The data used is called `sp100_2004-8`, which contains data from 2004 to 2008. After downloading the data, store the folder `sp100_2004-8` inside `data` as can be observed above in [Project Setup](#setup).
Then, download from [this drive](https://drive.google.com/file/d/1SrcHEkzGBf8P73Pl8ty7U_4VPUHuJvfm/view?usp=sharing) the file `raw_full_bbo_data.parquet` and store it into the folder `data` you have created.

## Team Components
For any question and/or curiosity, feel free to reach
* [Valentin Aolaritei](mailto:valentin.aolaritei@epfl.ch)
* [Alberto De Laurentis](mailto:alberto.delaurentis@epfl.ch)
* [Giorgio Milani](mailto:giorgio.milani@epfl.ch)
