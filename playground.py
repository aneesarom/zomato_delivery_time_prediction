import time
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
import pandas as pd

driver = webdriver.Chrome(options=Options())
actions = ActionChains(driver)

df = pd.read_csv("notebooks/data/finalTrain.csv")
res_lat_list = df["Restaurant_latitude"].to_list()
res_long_list = df["Restaurant_longitude"].to_list()
delivery_location_latitude_list = df["Delivery_location_latitude"].to_list()
delivery_location_longitude_list = df["Delivery_location_longitude"].to_list()

time_taken_list = []
km_list = []

for i in range(len(res_lat_list)):
    res_lat = res_lat_list[i]
    res_long = res_long_list[i]
    del_lat = delivery_location_latitude_list[i]
    del_long = delivery_location_longitude_list[i]
    print(res_lat, res_long, del_long, del_lat)
    try:
        driver.get(f"https://www.google.com/maps/dir/{res_lat},+{res_long}/{del_lat},+{del_long}/")
        time.sleep(5)
        hours = driver.find_element(By.CLASS_NAME, "MespJc")
        details = hours.text
        try:
            time.sleep(2)
            time_taken = details.split("\n")[0].split(" ")[0]
            time_taken_list.append(time_taken)
        except Exception as err:
            time_taken_list.append(np.NAN)
        try:
            time.sleep(2)
            km = details.split(" ")[1].split("\n")[1]
            km_list.append(km)
        except Exception as err:
            km_list.append(np.NAN)
    except Exception as err:
        time_taken_list.append(np.NAN)
        km_list.append(np.NAN)
    print(time_taken_list, km_list)
    time.sleep(2)

df["actual_time"] = time_taken_list
df["actual_km"] = km_list
df.to_csv("updated.csv")