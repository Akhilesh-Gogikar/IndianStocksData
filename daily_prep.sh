#!/bin/bash

# Run each Python script sequentially

python3 /home/a/Fin\ Project/Algo_Trading_Using_Breeze_API/Autologin.py
if [ $? -ne 0 ]; then
  echo "BSE data downloader failed to execute"
  exit 1
fi

python3 /home/a/Fin\ Project/Financial\ Web\ Scraping/BSE_data_downloader.py
if [ $? -ne 0 ]; then
  echo "BSE data downloader failed to execute"
  exit 1
fi

python3 /home/a/Fin\ Project/Financial\ Web\ Scraping/TickerTape_Scraper.py
if [ $? -ne 0 ]; then
  echo "TickerTape Scraper failed to execute"
  exit 1
fi

python3 /home/a/Fin\ Project/Financial\ Web\ Scraping/concurrentnews_data_extractor.py
if [ $? -ne 0 ]; then
  echo "Concurrent News Data Extractor failed to execute"
  exit 1
fi

python3 /home/a/Fin\ Project/Financial\ Web\ Scraping/fetch_latest_data.py
if [ $? -ne 0 ]; then
  echo "Latest Data Extractor failed to execute"
  exit 1
fi

# Add more scripts as needed

# Log completion
echo "All scripts executed successfully on $(date)" >> logfile.log
