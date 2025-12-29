"""
Core app views.
"""

from django.http import JsonResponse
from django.shortcuts import render

from .models import Tenant


def health_check(request):
    """Health check endpoint."""
    return JsonResponse({
        "status": "healthy",
        "service": "adk-flow-platform",
    })


def chat_tester(request):
    """Serve the WhatsApp-like chat tester UI."""
    return render(request, "chat.html")


def list_tenants(request):
    """List all tenants (for the chat tester dropdown)."""
    tenants = Tenant.objects.filter(is_active=True)
    return JsonResponse({
        "tenants": [
            {
                "id": str(t.id),
                "name": t.name,
                "slug": t.slug,
            }
            for t in tenants
        ]
    })
