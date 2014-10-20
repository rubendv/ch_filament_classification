import sys
print(sys.path)
import web.settings
import django
django.setup()
from django.contrib.auth.models import User

u = User(username='admin')
u.set_password('admin')
u.is_superuser = True
u.is_staff = True
u.save()
