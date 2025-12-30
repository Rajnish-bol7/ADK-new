"""
Flow models for the ADK Flow Platform.

This module contains models for storing flow definitions and execution sessions.
"""

import uuid
from django.db import models
from django.conf import settings


class Flow(models.Model):
    """
    Represents a flow definition created by a user.
    
    The flow_json field stores the complete flow structure including
    nodes, connections, and settings from the frontend flow builder.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        related_name="flows",
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name="created_flows",
    )
    
    # Flow metadata
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, default="")
    
    # Flow definition (JSON from frontend)
    flow_json = models.JSONField(default=dict)
    
    # Status
    is_active = models.BooleanField(default=True)
    is_published = models.BooleanField(default=False)
    
    # Versioning
    version = models.IntegerField(default=1)
    
    # Trigger settings
    TRIGGER_TYPES = [
        ("manual", "Manual"),
        ("chat", "Chat"),
        ("call", "Voice Call"),
        ("webhook", "Webhook"),
        ("scheduled", "Scheduled"),
    ]
    trigger_type = models.CharField(
        max_length=20,
        choices=TRIGGER_TYPES,
        default="chat",
    )
    
    # Webhook settings (if trigger_type is webhook)
    webhook_path = models.CharField(max_length=255, blank=True, default="")
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    published_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        db_table = "flows"
        ordering = ["-updated_at"]
        unique_together = [["tenant", "name"]]
    
    def __str__(self):
        return f"{self.name} ({self.tenant.name})"
    
    def get_flow_schema(self):
        """Convert to FlowSchema for ADK integration."""
        from adk_integration.schema.flow_schema import FlowSchema
        
        # Exclude fields that are already passed explicitly to avoid duplicate keyword arguments
        flow_json_clean = {k: v for k, v in self.flow_json.items() 
                          if k not in ["id", "flow_name", "description", "tenant_id"]}
        
        return FlowSchema(
            tenant_id=str(self.tenant.id),
            flow_id=str(self.id),
            flow_name=self.name,
            description=self.description,
            is_active=self.is_active,
            version=self.version,
            **flow_json_clean,
        )


class FlowVersion(models.Model):
    """
    Stores historical versions of flows for rollback capability.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    flow = models.ForeignKey(
        Flow,
        on_delete=models.CASCADE,
        related_name="versions",
    )
    
    version = models.IntegerField()
    flow_json = models.JSONField(default=dict)
    
    # Who made this version
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
    )
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = "flow_versions"
        ordering = ["-version"]
        unique_together = [["flow", "version"]]
    
    def __str__(self):
        return f"{self.flow.name} v{self.version}"


class Session(models.Model):
    """
    Represents a conversation session with a flow.
    
    Each session maintains state across multiple interactions with an agent.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    flow = models.ForeignKey(
        Flow,
        on_delete=models.CASCADE,
        related_name="sessions",
    )
    
    # User identification (can be authenticated user or anonymous)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="flow_sessions",
    )
    external_user_id = models.CharField(max_length=255, blank=True, default="")
    
    # Session state
    state = models.JSONField(default=dict)
    
    # Status
    STATUS_CHOICES = [
        ("active", "Active"),
        ("completed", "Completed"),
        ("expired", "Expired"),
        ("error", "Error"),
    ]
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default="active",
    )
    
    # Metadata
    metadata = models.JSONField(default=dict)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        db_table = "sessions"
        ordering = ["-updated_at"]
    
    def __str__(self):
        return f"Session {self.id} ({self.flow.name})"


class Message(models.Model):
    """
    Stores messages exchanged in a session.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(
        Session,
        on_delete=models.CASCADE,
        related_name="messages",
    )
    
    # Message content
    ROLE_CHOICES = [
        ("user", "User"),
        ("assistant", "Assistant"),
        ("system", "System"),
        ("tool", "Tool"),
    ]
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    content = models.TextField()
    
    # Tool-related fields
    tool_name = models.CharField(max_length=100, blank=True, default="")
    tool_call_id = models.CharField(max_length=100, blank=True, default="")
    
    # Metadata (tokens used, latency, etc.)
    metadata = models.JSONField(default=dict)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = "messages"
        ordering = ["created_at"]
    
    def __str__(self):
        return f"{self.role}: {self.content[:50]}..."


class FlowExecution(models.Model):
    """
    Logs individual flow executions for analytics and debugging.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    flow = models.ForeignKey(
        Flow,
        on_delete=models.CASCADE,
        related_name="executions",
    )
    session = models.ForeignKey(
        Session,
        on_delete=models.SET_NULL,
        null=True,
        related_name="executions",
    )
    
    # Execution details
    STATUS_CHOICES = [
        ("running", "Running"),
        ("completed", "Completed"),
        ("failed", "Failed"),
        ("timeout", "Timeout"),
    ]
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default="running",
    )
    
    # Input/Output
    input_message = models.TextField()
    output_message = models.TextField(blank=True, default="")
    
    # Error tracking
    error_message = models.TextField(blank=True, default="")
    error_traceback = models.TextField(blank=True, default="")
    
    # Metrics
    tokens_input = models.IntegerField(default=0)
    tokens_output = models.IntegerField(default=0)
    latency_ms = models.IntegerField(default=0)
    
    # Timestamps
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        db_table = "flow_executions"
        ordering = ["-started_at"]
    
    def __str__(self):
        return f"Execution {self.id} ({self.status})"


