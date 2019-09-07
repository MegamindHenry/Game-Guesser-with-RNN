import requests
import json

resp = requests.get("http://api.steampowered.com/ISteamApps/GetAppList/v2")

if resp.status_code != 200:
    resp.raise_for_status()

resp_json = resp.json()
apps_id = resp_json["applist"]["apps"]

output = json.dumps(apps_id, indent=4)

with open('games_remain_list.json', 'w+') as fp:
    fp.write(output)