# fix_feature_view.py
# Creates a new Feature View v2 pointing to v2 online-enabled feature groups
# Run from services/price-predictor/ folder

import hopsworks
from config import hopsworks_credentials as hw_config

project = hopsworks.login(
    host=hw_config.hopsworks_host,
    project=hw_config.project_name,
    api_key_value=hw_config.api_key,
)
fs = project.get_feature_store()

# Get v2 feature groups (online enabled)
tech_fg = fs.get_feature_group('technical_indicators', version=2)
news_fg = fs.get_feature_group('news_signals', version=2)

print(f'tech_fg online_enabled: {tech_fg.online_enabled}')
print(f'news_fg online_enabled: {news_fg.online_enabled}')

# Create Feature View v2 joining v2 groups
query = (
    tech_fg.select_all()
    .join(
        news_fg.select_all(),
        on=['coin'],
        join_type='left',
        prefix='news_signals_',
    )
    .filter((tech_fg.candle_seconds == 60))
)

feature_view = fs.create_feature_view(
    name='price_predictor',
    version=2,
    query=query,
)

print('Feature View v2 created successfully')
print(f'Name: {feature_view.name}, Version: {feature_view.version}')
