"""
Admin configuration for core models.
"""

from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import Tenant, User, APIKey


@admin.register(Tenant)
class TenantAdmin(admin.ModelAdmin):
    list_display = ["name", "slug", "is_active", "max_flows", "created_at"]
    list_filter = ["is_active", "created_at"]
    search_fields = ["name", "slug"]
    prepopulated_fields = {"slug": ("name",)}
    readonly_fields = ["id", "created_at", "updated_at"]
    
    fieldsets = (
        (None, {
            "fields": ("id", "name", "slug", "is_active")
        }),
        ("API Keys", {
            "fields": ("google_api_key", "openai_api_key", "anthropic_api_key"),
            "classes": ("collapse",),
        }),
        ("Limits", {
            "fields": ("max_flows", "max_requests_per_minute"),
        }),
        ("Timestamps", {
            "fields": ("created_at", "updated_at"),
            "classes": ("collapse",),
        }),
    )


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    list_display = ["username", "email", "tenant", "is_staff", "is_active"]
    list_filter = ["is_staff", "is_active", "tenant"]
    search_fields = ["username", "email"]
    
    fieldsets = BaseUserAdmin.fieldsets + (
        ("Tenant", {
            "fields": ("tenant", "phone", "avatar"),
        }),
    )
    
    add_fieldsets = BaseUserAdmin.add_fieldsets + (
        ("Tenant", {
            "fields": ("tenant",),
        }),
    )


@admin.register(APIKey)
class APIKeyAdmin(admin.ModelAdmin):
    list_display = ["name", "user", "is_active", "last_used_at", "created_at"]
    list_filter = ["is_active", "created_at"]
    search_fields = ["name", "user__email"]
    readonly_fields = ["id", "key", "created_at", "last_used_at"]
