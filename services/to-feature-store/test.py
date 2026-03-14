import datetime

import hopsworks
import pandas as pd

PROJECT = 'hritik_projects'
API_KEY = (
    'HxBGPZyTJOTqDQHM.aHZrZTshAnPAS2p5WzRkIIMAs5HP8YaugAAADERlJMBPtMPZbD27jueEhDOOyzwW'
)
FG_NAME = 'technical_indicators'
FG_VERSION = 2
# TARGET_TS = 1766918755990
TARGET_TS = 1764321359777

# def main():
#     project = hopsworks.login(project=PROJECT, api_key_value=API_KEY)
#     fs = project.get_feature_store()

#     fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)

#     # Read dataframe (older hsfs doesn't support columns=...)
#     df = fg.read(online=False)

#     latest_ts = int(df["timestamp_ms"].min())

#     print("Latest timestamp_ms:", latest_ts)
#     print("Latest time:", datetime.datetime.fromtimestamp(latest_ts / 1000))

# if __name__ == "__main__":
#     main()


def main():
    project = hopsworks.login(project=PROJECT, api_key_value=API_KEY)
    fs = project.get_feature_store()

    fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)

    print('Reading offline feature group (can take time)...')
    df = fg.read(online=False)

    print(f'Total rows read: {len(df)}')

    rows = df[df['timestamp_ms'] == TARGET_TS]

    if rows.empty:
        print(f'❌ No rows found for timestamp_ms = {TARGET_TS}')
        return

    print(f'\n✅ Found {len(rows)} row(s) for timestamp_ms = {TARGET_TS}')
    print('row: ', rows)
    print(f'Human time: {datetime.fromtimestamp(TARGET_TS / 1000)}\n')

    # Print ALL columns for those rows
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)

    print(rows)


if __name__ == '__main__':
    main()
