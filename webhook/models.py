from django.db import models


class WhatsAppMessage(models.Model):
    """Model to store incoming WhatsApp messages"""
    
    MESSAGE_TYPES = [
        ('text', 'Text'),
        ('audio', 'Audio'),
        ('image', 'Image'),
        ('video', 'Video'),
        ('document', 'Document'),
        ('location', 'Location'),
        ('contacts', 'Contacts'),
        ('sticker', 'Sticker'),
    ]
    
    # Message identification
    message_id = models.CharField(max_length=255, unique=True)
    wa_id = models.CharField(max_length=50, help_text="WhatsApp ID of the sender")
    from_number = models.CharField(max_length=50, help_text="Phone number of the sender")
    
    # Contact information
    contact_name = models.CharField(max_length=255, blank=True, null=True)
    
    # Message details
    message_type = models.CharField(max_length=20, choices=MESSAGE_TYPES)
    message_text = models.TextField(blank=True, null=True, help_text="Text content if message type is text")
    
    # Audio specific fields (for voice notes)
    audio_id = models.CharField(max_length=255, blank=True, null=True)
    audio_url = models.URLField(blank=True, null=True)
    audio_mime_type = models.CharField(max_length=100, blank=True, null=True)
    is_voice = models.BooleanField(default=False)

    # Image specific fields
    image_id = models.CharField(max_length=255, blank=True, null=True)
    image_url = models.URLField(blank=True, null=True)
    image_mime_type = models.CharField(max_length=100, blank=True, null=True)
    image_caption = models.TextField(blank=True, null=True)

    # Video specific fields
    video_id = models.CharField(max_length=255, blank=True, null=True)
    video_url = models.URLField(blank=True, null=True)
    video_mime_type = models.CharField(max_length=100, blank=True, null=True)
    video_caption = models.TextField(blank=True, null=True)

    # Document specific fields
    document_id = models.CharField(max_length=255, blank=True, null=True)
    document_url = models.URLField(blank=True, null=True)
    document_filename = models.CharField(max_length=255, blank=True, null=True)
    document_mime_type = models.CharField(max_length=100, blank=True, null=True)

    # Location specific fields
    latitude = models.DecimalField(max_digits=10, decimal_places=8, blank=True, null=True)
    longitude = models.DecimalField(max_digits=11, decimal_places=8, blank=True, null=True)

    # Sticker specific fields
    sticker_id = models.CharField(max_length=255, blank=True, null=True)
    sticker_url = models.URLField(blank=True, null=True)
    sticker_mime_type = models.CharField(max_length=100, blank=True, null=True)
    is_animated = models.BooleanField(default=False)

    # Contacts specific fields (stored as JSON)
    contacts_data = models.JSONField(blank=True, null=True, help_text="Contact information in JSON format")
    
    # Flow reference (which flow processed this message)
    flow = models.ForeignKey(
        'flows.Flow',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='whatsapp_messages',
        help_text="The flow that received this message via webhook"
    )
    
    # Metadata
    timestamp = models.CharField(max_length=50)
    phone_number_id = models.CharField(max_length=100, help_text="Meta's phone number ID")
    display_phone_number = models.CharField(max_length=50, help_text="Display phone number")
    
    # Raw payload for reference
    raw_payload = models.JSONField(default=dict, help_text="Complete webhook payload")
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False, help_text="Whether this message has been processed by chatbot")
    
    class Meta:
        verbose_name = 'WhatsApp Message'
        verbose_name_plural = 'WhatsApp Messages'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['wa_id', '-created_at']),
            models.Index(fields=['message_type']),
            models.Index(fields=['processed']),
            models.Index(fields=['flow', '-created_at']),
        ]
    
    def __str__(self):
        return f"{self.from_number} - {self.message_type} - {self.created_at}"


class WhatsAppCall(models.Model):
    """Model to store incoming WhatsApp call events (each event stored separately)"""
    
    CALL_EVENTS = [
        ('connect', 'Connect'),
        ('terminate', 'Terminate'),
    ]
    
    CALL_DIRECTIONS = [
        ('USER_INITIATED', 'User Initiated'),
        ('BUSINESS_INITIATED', 'Business Initiated'),
    ]
    
    CALL_STATUSES = [
        ('COMPLETED', 'Completed'),
        ('MISSED', 'Missed'),
        ('DECLINED', 'Declined'),
        ('FAILED', 'Failed'),
        ('BUSY', 'Busy'),
        ('NO_ANSWER', 'No Answer'),
    ]
    
    # Call identification
    call_id = models.CharField(max_length=255, help_text="Unique call ID from WhatsApp")
    from_number = models.CharField(max_length=50, help_text="Phone number of the caller")
    to_number = models.CharField(max_length=50, help_text="Phone number being called")
    
    # Contact information
    wa_id = models.CharField(max_length=50, help_text="WhatsApp ID of the caller")
    contact_name = models.CharField(max_length=255, blank=True, null=True)
    
    # Call details
    event = models.CharField(max_length=20, choices=CALL_EVENTS, help_text="Call event type: connect or terminate")
    direction = models.CharField(max_length=30, choices=CALL_DIRECTIONS, help_text="Who initiated the call")
    status = models.CharField(max_length=20, choices=CALL_STATUSES, blank=True, null=True, help_text="Call status (available on terminate)")
    
    # Call timing (available on completed/terminate events)
    timestamp = models.CharField(max_length=50, help_text="Event timestamp")
    start_time = models.CharField(max_length=50, blank=True, null=True, help_text="Call start time")
    end_time = models.CharField(max_length=50, blank=True, null=True, help_text="Call end time")
    duration = models.IntegerField(blank=True, null=True, help_text="Call duration in seconds")
    
    # Session data (for connect events - contains SDP for WebRTC)
    session_sdp = models.TextField(blank=True, null=True, help_text="SDP offer/answer for WebRTC call handling")
    session_sdp_type = models.CharField(max_length=20, blank=True, null=True, help_text="SDP type: offer or answer")
    
    # Metadata
    phone_number_id = models.CharField(max_length=100, help_text="Meta's phone number ID")
    display_phone_number = models.CharField(max_length=50, help_text="Display phone number")
    
    # Raw payload for reference
    raw_payload = models.JSONField(default=dict, help_text="Complete call webhook payload")
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = 'WhatsApp Call'
        verbose_name_plural = 'WhatsApp Calls'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['call_id']),
            models.Index(fields=['from_number', '-created_at']),
            models.Index(fields=['event']),
            models.Index(fields=['status']),
        ]
    
    def __str__(self):
        return f"{self.from_number} -> {self.to_number} | {self.event} | {self.status or 'N/A'} | {self.created_at}"


class WhatsAppMessageStatus(models.Model):
    """Model to store message status updates (sent/delivered/read/failed) for outgoing messages"""
    
    STATUS_TYPES = [
        ('sent', 'Sent'),
        ('delivered', 'Delivered'),
        ('read', 'Read'),
        ('failed', 'Failed'),
    ]
    
    # Status identification
    message_id = models.CharField(max_length=255, help_text="ID of the message this status refers to")
    status = models.CharField(max_length=20, choices=STATUS_TYPES, help_text="Message status")
    recipient_id = models.CharField(max_length=50, help_text="Phone number of the recipient")
    
    # Conversation information
    conversation_id = models.CharField(max_length=255, blank=True, null=True, help_text="Conversation ID")
    conversation_expiration_timestamp = models.CharField(max_length=50, blank=True, null=True, help_text="Conversation expiration timestamp")
    conversation_origin_type = models.CharField(max_length=50, blank=True, null=True, help_text="Origin type (service/user)")
    
    # Pricing information
    is_billable = models.BooleanField(default=False, help_text="Whether the message is billable")
    pricing_model = models.CharField(max_length=50, blank=True, null=True, help_text="Pricing model (e.g., PMP)")
    pricing_category = models.CharField(max_length=50, blank=True, null=True, help_text="Pricing category (e.g., service)")
    pricing_type = models.CharField(max_length=100, blank=True, null=True, help_text="Pricing type (e.g., free_customer_service)")
    
    # Metadata
    timestamp = models.CharField(max_length=50, help_text="Status timestamp")
    phone_number_id = models.CharField(max_length=100, help_text="Meta's phone number ID")
    display_phone_number = models.CharField(max_length=50, help_text="Display phone number")
    
    # Raw payload for reference
    raw_payload = models.JSONField(default=dict, help_text="Complete status payload")
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = 'WhatsApp Message Status'
        verbose_name_plural = 'WhatsApp Message Statuses'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['message_id', '-created_at']),
            models.Index(fields=['status']),
            models.Index(fields=['recipient_id', '-created_at']),
        ]
    
    def __str__(self):
        return f"{self.recipient_id} | {self.status} | Message: {self.message_id[:20]}... | {self.created_at}"


class WhatsAppOutgoingMessage(models.Model):
    """Model to track outgoing messages we send"""
    
    MESSAGE_TYPES = [
        ('text', 'Text'),
        ('template', 'Template'),
        ('image', 'Image'),
        ('video', 'Video'),
        ('document', 'Document'),
        ('audio', 'Audio'),
    ]
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('sent', 'Sent'),
        ('delivered', 'Delivered'),
        ('read', 'Read'),
        ('failed', 'Failed'),
    ]
    
    # Message identification
    message_id = models.CharField(max_length=255, unique=True, blank=True, null=True, help_text="WhatsApp message ID returned from API")
    to_number = models.CharField(max_length=50, help_text="Recipient phone number")
    
    # Message content
    message_type = models.CharField(max_length=20, choices=MESSAGE_TYPES, default='text')
    message_text = models.TextField(help_text="Message content")
    
    # Status tracking
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending', help_text="Current message status")
    api_response = models.JSONField(blank=True, null=True, help_text="Response from WhatsApp API")
    error_message = models.TextField(blank=True, null=True, help_text="Error message if sending failed")
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    sent_at = models.DateTimeField(blank=True, null=True, help_text="When message was successfully sent")
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'WhatsApp Outgoing Message'
        verbose_name_plural = 'WhatsApp Outgoing Messages'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['to_number', '-created_at']),
            models.Index(fields=['status']),
            models.Index(fields=['message_id']),
        ]
    
    def __str__(self):
        return f"{self.to_number} | {self.status} | {self.message_text[:30]}... | {self.created_at}"
