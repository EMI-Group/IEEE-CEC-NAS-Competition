# Generated by Django 3.2.12 on 2022-05-07 14:56

from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='MobileNetV3Result',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('index', models.TextField(db_index=True, max_length=256, null=True)),
                ('phenotype', models.JSONField(default=dict)),
                ('genotype', models.JSONField(default=dict)),
                ('result', models.JSONField(default=dict)),
                ('params', models.IntegerField()),
                ('flops', models.IntegerField()),
                ('latency', models.FloatField()),
                ('valid_acc', models.FloatField()),
                ('test_acc', models.FloatField()),
            ],
            options={
                'abstract': False,
            },
        ),
    ]