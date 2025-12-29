"""
Core models for the ADK Flow Platform.

This module contains the fundamental models for multi-tenant user management.
"""

import uuid
from django.contrib.auth.models import AbstractUser
from django.db import models


class Tenant(models.Model):
    """
    Represents a tenant (organization/company) in the multi-tenant system.
    
    Each tenant can have multiple users and flows.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    slug = models.SlugField(max_length=100, unique=True)
    
    # API Keys for different providers (encrypted in production)
    google_api_key = models.CharField(max_length=500, blank=True, default="")
    openai_api_key = models.CharField(max_length=500, blank=True, default="")
    anthropic_api_key = models.CharField(max_length=500, blank=True, default="")
    
    # Settings
    is_active = models.BooleanField(default=True)
    max_flows = models.IntegerField(default=100)
    max_requests_per_minute = models.IntegerField(default=60)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = "tenants"
        ordering = ["-created_at"]
    
    def __str__(self):
        return self.name
    
    def get_api_keys(self) -> dict:
        """Get all API keys for this tenant."""
        return {
            "google": self.google_api_key,
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
        }


class User(AbstractUser):
    """
    Custom user model with tenant association.
    
    Each user belongs to a tenant and inherits their API keys and settings.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(
        Tenant,
        on_delete=models.CASCADE,
        related_name="users",
        null=True,
        blank=True,
    )
    
    # Profile
    phone = models.CharField(max_length=20, blank=True, default="")
    avatar = models.ImageField(upload_to="avatars/", null=True, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = "users"
        ordering = ["-created_at"]
    
    def __str__(self):
        return self.email or self.username
    
    @property
    def tenant_id_str(self) -> str:
        """Get tenant ID as string for ADK integration."""
        return str(self.tenant.id) if self.tenant else ""


class APIKey(models.Model):
    """
    API keys for external access to the platform.
    
    Users can create API keys to access flows programmatically.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="api_keys",
    )
    name = models.CharField(max_length=100)
    key = models.CharField(max_length=64, unique=True)
    
    # Permissions
    can_read = models.BooleanField(default=True)
    can_write = models.BooleanField(default=False)
    can_execute = models.BooleanField(default=True)
    
    # Limits
    rate_limit = models.IntegerField(default=60)  # requests per minute
    
    # Status
    is_active = models.BooleanField(default=True)
    last_used_at = models.DateTimeField(null=True, blank=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = "api_keys"
        ordering = ["-created_at"]
    
    def __str__(self):
        return f"{self.name} ({self.user.email})"
