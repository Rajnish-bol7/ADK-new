from django.contrib import admin
from .models import WhatsAppMessage, WhatsAppCall, WhatsAppMessageStatus, WhatsAppOutgoingMessage


@admin.register(WhatsAppMessage)
class WhatsAppMessageAdmin(admin.ModelAdmin):
    list_display = ['from_number', 'contact_name', 'flow', 'message_type', 'message_text', 'created_at', 'processed']
    list_filter = ['message_type', 'processed', 'flow', 'created_at']
    search_fields = ['from_number', 'contact_name', 'message_text', 'message_id', 'flow__name']
    readonly_fields = ['message_id', 'wa_id', 'from_number', 'contact_name', 'message_type', 
                      'message_text', 'audio_id', 'audio_url', 'audio_mime_type', 'is_voice',
                      'timestamp', 'phone_number_id', 'display_phone_number', 'raw_payload', 
                      'flow', 'created_at']
    
    def has_add_permission(self, request):
        return False  # Messages are only created via webhook


@admin.register(WhatsAppCall)
class WhatsAppCallAdmin(admin.ModelAdmin):
    list_display = ['from_number', 'to_number', 'contact_name', 'event', 'status', 'direction', 'duration', 'created_at']
    list_filter = ['event', 'status', 'direction', 'created_at']
    search_fields = ['from_number', 'to_number', 'contact_name', 'call_id']
    readonly_fields = ['call_id', 'from_number', 'to_number', 'wa_id', 'contact_name', 
                      'event', 'direction', 'status', 'timestamp', 'start_time', 'end_time',
                      'duration', 'session_sdp', 'session_sdp_type', 'phone_number_id',
                      'display_phone_number', 'raw_payload', 'created_at']
    
    def has_add_permission(self, request):
        return False  # Calls are only created via webhook


@admin.register(WhatsAppMessageStatus)
class WhatsAppMessageStatusAdmin(admin.ModelAdmin):
    list_display = ['recipient_id', 'status', 'message_id', 'is_billable', 'pricing_type', 'created_at']
    list_filter = ['status', 'is_billable', 'pricing_category', 'created_at']
    search_fields = ['recipient_id', 'message_id', 'conversation_id']
    readonly_fields = ['message_id', 'status', 'recipient_id', 'conversation_id',
                      'conversation_expiration_timestamp', 'conversation_origin_type',
                      'is_billable', 'pricing_model', 'pricing_category', 'pricing_type',
                      'timestamp', 'phone_number_id', 'display_phone_number', 'raw_payload', 'created_at']
    
    def has_add_permission(self, request):
        return False  # Statuses are only created via webhook


@admin.register(WhatsAppOutgoingMessage)
class WhatsAppOutgoingMessageAdmin(admin.ModelAdmin):
    list_display = ['to_number', 'message_text', 'status', 'message_id', 'created_at', 'sent_at']
    list_filter = ['status', 'message_type', 'created_at']
    search_fields = ['to_number', 'message_text', 'message_id']
    readonly_fields = ['message_id', 'to_number', 'message_type', 'message_text', 'status',
                      'api_response', 'error_message', 'created_at', 'sent_at', 'updated_at']
    
    def has_add_permission(self, request):
        return False  # Outgoing messages are created via API endpoint only
