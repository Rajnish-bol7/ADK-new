from rest_framework import serializers


class WhatsAppWebhookSerializer(serializers.Serializer):
    """Serializer for validating WhatsApp webhook payload structure"""
    object = serializers.CharField()
    entry = serializers.ListField()

