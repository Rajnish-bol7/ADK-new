# Generated migration for adding react_flow_json field

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('flows', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='flow',
            name='react_flow_json',
            field=models.JSONField(blank=True, default=dict, null=True),
        ),
    ]

