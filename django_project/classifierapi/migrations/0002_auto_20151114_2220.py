# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('classifierapi', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Model',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('model_file', models.CharField(max_length=2000)),
            ],
        ),
        migrations.RemoveField(
            model_name='user',
            name='model_file',
        ),
        migrations.AddField(
            model_name='model',
            name='user',
            field=models.ForeignKey(to='classifierapi.User'),
        ),
    ]
