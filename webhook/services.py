"""
Service functions for WhatsApp Business API operations
"""
import requests
from django.conf import settings
import logging

logger = logging.getLogger(__name__)


def send_whatsapp_message(to_number, message_text, message_type='text'):
    """
    Send a message via WhatsApp Business API
    
    Args:
        to_number: Recipient phone number (with country code, e.g., 918279486865)
        message_text: Text content to send
        message_type: Type of message (text, image, etc.)
    
    Returns:
        dict: {'success': True/False, 'message_id': '...', 'error': '...', 'response': {...}}
    """
    if not settings.WHATSAPP_ACCESS_TOKEN or not settings.WHATSAPP_PHONE_NUMBER_ID:
        error_msg = "WhatsApp API credentials not configured. Please set WHATSAPP_ACCESS_TOKEN and WHATSAPP_PHONE_NUMBER_ID in .env"
        logger.error(error_msg)
        return {
            'success': False,
            'error': error_msg
        }
    
    url = f"{settings.WHATSAPP_API_BASE_URL}/{settings.WHATSAPP_PHONE_NUMBER_ID}/messages"
    
    headers = {
        'Authorization': f'Bearer {settings.WHATSAPP_ACCESS_TOKEN}',
        'Content-Type': 'application/json',
    }
    
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": message_type,
        "text": {
            "body": message_text
        }
    }
    
    try:
        logger.info(f"Attempting to send message to {to_number}")
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        message_id = data.get('messages', [{}])[0].get('id')
        
        logger.info(f"Message sent successfully: {message_id} to {to_number}")
        
        return {
            'success': True,
            'message_id': message_id,
            'response': data
        }
    
    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP Error: {e.response.status_code} - {e.response.text}"
        logger.error(f"Failed to send message: {error_msg}")
        return {
            'success': False,
            'error': error_msg,
            'response': e.response.json() if e.response.content else None
        }
    
    except requests.exceptions.RequestException as e:
        error_msg = f"Request Error: {str(e)}"
        logger.error(f"Failed to send message: {error_msg}")
        return {
            'success': False,
            'error': error_msg
        }
    
    except Exception as e:
        error_msg = f"Unexpected Error: {str(e)}"
        logger.error(f"Failed to send message: {error_msg}", exc_info=True)
        return {
            'success': False,
            'error': error_msg
        }

