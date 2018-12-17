import requests
import json


for lp in range(10):
    with open('app_info.json', 'r') as f:
        app_all = json.load(f)

    print(len(app_all))
    # print(app_all)

    resp = requests.get("http://api.steampowered.com/ISteamApps/GetAppList/v2")
    if resp.status_code != 200:
        # This means something went wrong.
        resp.raise_for_status()
    resp_json = resp.json()
    # print(resp_json)
    apps = resp_json["applist"]["apps"]

    rest_apps = [x for x in apps if str(x["appid"]) not in app_all]

    for app in rest_apps[:20]:
        app_resp = requests.get("http://store.steampowered.com/api/appdetails/?appids=" + str(app["appid"]))
        if app_resp.status_code != 200:
            # This means something went wrong.
            app_resp.raise_for_status()
        app_resp_json = app_resp.json()
        print(app_resp_json)
        app_all.update(app_resp_json)


    del_key = []
    for key in app_all:
        for k, v in app_all[key].items():
            if k == "success" and v == False:
                del_key.append(key)
    for key in del_key:
        del app_all[key]

    print(len(app_all))

    output_all = json.dumps(app_all, indent=4)

    with open('app_info.json', 'w+') as f:
        f.write(output_all)
