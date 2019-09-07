import requests
import json
import time
for _ in range(100):
    try:
        with open('games.json', 'r') as fp:
            apps_data = json.load(fp)
    except:
        print('No game file')
        apps_data = {}

    try:
        with open('games_remain_list.json', 'r') as fp:
            apps_remain_list = json.load(fp)
    except:
        print('No remain file')
        apps_remain_list = []

    try:
        with open('games_failure_list.json', 'r') as fp:
            apps_failure = json.load(fp)
    except:
        print('No failure file')
        apps_failure = []

    print('remain: {}'.format(len(apps_remain_list)))
    print('load: {}'.format(len(apps_data)))
    print('failure: {}'.format(len(apps_failure)))

    query_games = apps_remain_list[:200]
    apps_remain_list = apps_remain_list[200:]
    for app in query_games:
        single_app_resp = requests.get("http://store.steampowered.com/api/appdetails/?appids=" + str(app["appid"]))

        if single_app_resp.status_code != 200:
            print('Failure: {}'.format(single_app_resp.status_code))
            # apps_failure.append(app)
            time.sleep(100)
            continue

        resp_json = single_app_resp.json()
        item = next(iter(resp_json.values()))

        if not item['success']:
            print('Failure: {}'.format(resp_json))
            apps_failure.append(app)
            continue

        # print('Success: {}'.format('jjjj'))
        apps_data.update(resp_json)

    output_games = json.dumps(apps_data, indent=4)
    output_failure = json.dumps(apps_failure, indent=4)
    output_remain = json.dumps(apps_remain_list, indent=4)

    with open('games.json', 'w+') as fp:
        fp.write(output_games)

    with open('games_failure_list.json', 'w+') as fp:
        fp.write(output_failure)

    with open('games_remain_list.json', 'w+') as fp:
        fp.write(output_remain)
