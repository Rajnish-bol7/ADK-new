"""
Admin configuration for flow models.
"""

from django.contrib import admin
from .models import Flow, FlowVersion, Session, Message, FlowExecution, Webhook


@admin.register(Flow)
class FlowAdmin(admin.ModelAdmin):
    list_display = ["name", "tenant", "trigger_type", "is_active", "is_published", "version", "updated_at"]
    list_filter = ["is_active", "is_published", "trigger_type", "tenant"]
    search_fields = ["name", "description"]
    readonly_fields = ["id", "created_at", "updated_at", "published_at"]
    
    fieldsets = (
        (None, {
            "fields": ("id", "tenant", "created_by", "name", "description")
        }),
        ("Flow Definition", {
            "fields": ("flow_json",),
            "classes": ("collapse",),
        }),
        ("Settings", {
            "fields": ("trigger_type", "webhook_path", "is_active", "is_published", "version"),
        }),
        ("Timestamps", {
            "fields": ("created_at", "updated_at", "published_at"),
            "classes": ("collapse",),
        }),
    )


@admin.register(FlowVersion)
class FlowVersionAdmin(admin.ModelAdmin):
    list_display = ["flow", "version", "created_by", "created_at"]
    list_filter = ["created_at"]
    search_fields = ["flow__name"]
    readonly_fields = ["id", "created_at"]


@admin.register(Session)
class SessionAdmin(admin.ModelAdmin):
    list_display = ["id", "flow", "user", "status", "created_at", "updated_at"]
    list_filter = ["status", "created_at"]
    search_fields = ["id", "flow__name"]
    readonly_fields = ["id", "created_at", "updated_at"]


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ["id", "session", "role", "content_preview", "created_at"]
    list_filter = ["role", "created_at"]
    search_fields = ["content"]
    readonly_fields = ["id", "created_at"]
    
    def content_preview(self, obj):
        return obj.content[:50] + "..." if len(obj.content) > 50 else obj.content
    content_preview.short_description = "Content"


@admin.register(FlowExecution)
class FlowExecutionAdmin(admin.ModelAdmin):
    list_display = ["id", "flow", "status", "tokens_input", "tokens_output", "latency_ms", "started_at"]
    list_filter = ["status", "started_at"]
    search_fields = ["flow__name"]
    readonly_fields = ["id", "started_at", "completed_at"]


@admin.register(Webhook)
class WebhookAdmin(admin.ModelAdmin):
    list_display = ["path", "flow", "auth_type", "is_active", "last_triggered_at"]
    list_filter = ["is_active", "auth_type"]
    search_fields = ["path", "flow__name"]
    readonly_fields = ["id", "created_at", "last_triggered_at"]
