import requests
import json
from random import shuffle
import time

start = time.time()
# count
error_count = 0
iter_count = 0

# get all appids from steam
resp = requests.get("http://api.steampowered.com/ISteamApps/GetAppList/v2")

# raise error if
if resp.status_code != 200:
    resp.raise_for_status()
    # print("requests error " + str(error_count))
    # error_count += 1
    # print(resp.status_code)
resp_json = resp.json()
apps = resp_json["applist"]["apps"]

# iterate as many times as we want
for lp in range(50):
    iter_count += 1
    print("iter:")
    print(iter_count)

    # open and load file
    with open('app_info.json', 'r') as f:
        app_all = json.load(f)

    print("numbers of items in:")
    print(len(app_all))

    # shuffle list to prevent bad appid
    shuffle(apps)
    # add list of apps is not in
    rest_apps = [x for x in apps[:20] if str(x["appid"]) not in app_all]

    # for each app query from store
    for app in rest_apps:
        app_resp = requests.get("http://store.steampowered.com/api/appdetails/?appids=" + str(app["appid"]))
        # just continue if wrong
        if app_resp.status_code != 200:
            # This means something went wrong.
            # app_resp.raise_for_status()
            print("requests error " + str(error_count))
            error_count += 1
            print(app_resp.status_code)
            continue
        app_resp_json = app_resp.json()
        print(app_resp_json)
        app_all.update(app_resp_json)

    # delete item if failure
    del_key = []
    for key in app_all:
        for k, v in app_all[key].items():
            if k == "success" and v == False:
                del_key.append(key)
    for key in del_key:
        del app_all[key]

    print("all lenth:")
    print(len(app_all))
    print("failure:")
    print(len(del_key))

    # format json to string
    output_all = json.dumps(app_all, indent=4)

    with open('app_info.json', 'w+') as f:
        f.write(output_all)

end = time.time()

print("running time:")
print(end - start)
