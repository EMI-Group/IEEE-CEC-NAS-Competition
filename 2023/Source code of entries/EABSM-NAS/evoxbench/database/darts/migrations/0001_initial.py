# Generated by Django 3.2.12 on 2022-03-28 03:17

from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='NASBench301Result',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('index', models.TextField(db_index=True, max_length=256, null=True)),
                ('phenotype', models.JSONField(default=dict)),
                ('genotype', models.JSONField(default=dict)),
                ('result', models.JSONField(default=dict)),
                ('normal', models.CharField(max_length=128)),
                ('normal_concat', models.CharField(max_length=128)),
                ('reduce', models.CharField(max_length=128)),
                ('reduce_concat', models.CharField(max_length=128)),
                ('epochs', models.IntegerField()),
                ('dataset_id', models.IntegerField()),
                ('dataset_path', models.CharField(max_length=256)),
            ],
            options={
                'abstract': False,
            },
        ),
    ]