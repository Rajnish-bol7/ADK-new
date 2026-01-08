"""
Admin configuration for flow models.
"""

import json
from django import forms
from django.contrib import admin
from .models import Flow, FlowVersion, Session, Message, FlowExecution


class PrettyJSONWidget(forms.Textarea):
    """Custom widget to display JSON in a formatted way."""
    
    def render(self, name, value, attrs=None, renderer=None):
        if value is None:
            value = "{}"
        
        # If it's a dict/list, format it
        if isinstance(value, (dict, list)):
            formatted_json = json.dumps(value, indent=2, ensure_ascii=False)
        else:
            # Try to parse and format if it's a string
            try:
                parsed = json.loads(value) if isinstance(value, str) else value
                formatted_json = json.dumps(parsed, indent=2, ensure_ascii=False)
            except (TypeError, ValueError):
                formatted_json = str(value) if value else "{}"
        
        # Create a styled textarea with formatted JSON
        attrs = attrs or {}
        attrs.update({
            'rows': 30,
            'cols': 120,
            'style': 'font-family: monospace; font-size: 12px; white-space: pre; overflow-x: auto;',
            'readonly': True,
        })
        
        return super().render(name, formatted_json, attrs, renderer)


@admin.register(Flow)
class FlowAdmin(admin.ModelAdmin):
    list_display = ["name", "tenant", "trigger_type", "is_active", "is_published", "version", "updated_at"]
    list_filter = ["is_active", "is_published", "trigger_type", "tenant"]
    search_fields = ["name", "description"]
    readonly_fields = ["id", "created_at", "updated_at", "published_at"]
    
    def get_form(self, request, obj=None, **kwargs):
        form = super().get_form(request, obj, **kwargs)
        # Use PrettyJSONWidget for JSON fields
        form.base_fields['flow_json'].widget = PrettyJSONWidget()
        form.base_fields['react_flow_json'].widget = PrettyJSONWidget()
        return form
    
    fieldsets = (
        (None, {
            "fields": ("id", "tenant", "created_by", "name", "description")
        }),
        ("Flow Definition (n8n format - for sharing/portability)", {
            "fields": ("flow_json",),
            "classes": ("collapse",),
            "description": "This is the n8n-like format, suitable for sharing and portability. This is what gets exported/shared with partners.",
        }),
        ("Flow Definition (React Flow format - for UI editing)", {
            "fields": ("react_flow_json",),
            "classes": ("collapse",),
            "description": "This is the original React Flow format used by the frontend UI for editing.",
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


