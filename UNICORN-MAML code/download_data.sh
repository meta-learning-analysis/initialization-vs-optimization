wget --header="Host: doc-0o-0c-docs.googleusercontent.com" --header="User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7" --header="Accept-Language: en-GB,en-US;q=0.9,en;q=0.8" --header="Referer: https://drive.google.com/" --header="Cookie: AUTH_j0lp5uqcmr9jp5fg3f87vsvfiauar9ha_nonce=7ibha84e1cue0" --header="Connection: keep-alive" "https://doc-0o-0c-docs.googleusercontent.com/docs/securesc/ph6mu5qd6l5pt1m2l6snlikeupib3nti/9i6rg9u7km8o4vp0n4a2pcm6qq1qe9p5/1687875525000/01924853128218817746/04111443461447131543/1nVGCTd9ttULRXFezh4xILQ9lUkg0WZCG?e=download&ax=ADWCPKC6g8d8auZoG8v4nCIopywjMXHI4rrTGGXnXsJc3eO5_qb1W_l9cP1Ll_OWmILCYl6Udii1kMvf9Ao5gx1ckkebnEyutPcybmj8eIPhpSaLPMGuBYfb1rtGzbmyAHIqtv4UGbUByYXLxtwAJ-IImzp_GyfHxXgwJ-2ItYDmRNKF0CbRxyQNXo3HZkIgrJWseUf_T_P7A-ii-hqPt5uXiitueGzxsGHwq_WDDwgo5LoOBeu2eDwGTU7-mLtKeJuhS4dbtCyyaVQklaz94DDL5N3MAduLNsRrGi8iBJkTRlInzDVuf5CpzCvfCbrepdfPHhyQxadVyBs-Ypypg-mSo93-Ye2sGO7HlUBVyZFuyoUsyAeMcTMt1_Co6MiCjSjmFLvukItiQjelCGQDKYZUiNYGQCLAdsadLYqzfPscIzIFg7iPMHNjR21Ko_Ky0B6frElWahWs5CqcWkr2dbvcVFVrhfWGZ6fPoYgIBgSDZ1tajR7gSVAA5wyjz_ZDriWi92HpLd0l0aZWvdACHKulnZ1Jy6F_MCJ9mSW2FwQ1upjq4trfS4nNJqedR_wcdW9qmUn51hYUg56USnVyqAOr1vOY-XaNuRsFv4OnEYl1d8P7hag-Nsxo2xllXHyiOiL9thcBBOH8g95hcugh337q2s6NPELAUyyz-FpJt5eTPIIdBMcIbCDIp50PpWmbsFKf1zgUnmgXSssVGIMid8ej7DRiQoC4CCaGG6Pl4GQV81yAn4ZSTPPT4UGdDdpCQrynUieALSDRh_N27Y-kol4cjL9l_Crz2OS71vqGhaY6826tsjLR2Q9xkTr8wbdMDJAd6hKaUoiKdJRo8AzC2WnrSGwqlyBmtGtd__nv2zRvAh6Qtdjn9I1covHA7dO8J3QDJPes400dqXKW_OMq5pO-9vRfbkjhosHzVIpZ9Oy_VB-HGbmvz4sGonWFIFm9pLZOvap0qyoUQeI&uuid=b5075508-e3c0-448f-968c-a098905d7653&authuser=0&nonce=7ibha84e1cue0&user=04111443461447131543&hash=g6dm22iubjagbgae9cnpgnoltr2gmcgo" -c -O 'tiered-imagenet-kwon.zip'
mv tiered-imagenet-kwon.zip tiered-imagenet.zip
DATA_ROOT="./data"
mkdir -p $DATA_ROOT/tiered-imagenet
cd $DATA_ROOT/tiered-imagenet
cp ../../tiered-imagenet.zip .
unzip tiered-imagenet.zip
rm -f tiered-imagenet.zip