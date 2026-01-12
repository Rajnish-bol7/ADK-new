"""
Utility functions for WhatsApp webhook processing
"""


def extract_message_content(message_data):
    """
    Extract message content based on message type.
    
    Args:
        message_data: The message object from webhook payload
    
    Returns:
        dict: Extracted message content
    """
    message_type = message_data.get('type')
    content = {
        'type': message_type,
        'text': None,
        'audio': None,
        'image': None,
        'video': None,
        'document': None,
    }
    
    if message_type == 'text':
        content['text'] = message_data.get('text', {}).get('body', '')
    
    elif message_type == 'audio':
        audio_data = message_data.get('audio', {})
        content['audio'] = {
            'id': audio_data.get('id'),
            'url': audio_data.get('url'),
            'mime_type': audio_data.get('mime_type'),
            'voice': audio_data.get('voice', False),
            'sha256': audio_data.get('sha256'),
        }
    
    elif message_type == 'image':
        image_data = message_data.get('image', {})
        content['image'] = {
            'id': image_data.get('id'),
            'url': image_data.get('url'),
            'mime_type': image_data.get('mime_type'),
            'caption': image_data.get('caption'),
        }
    
    elif message_type == 'video':
        video_data = message_data.get('video', {})
        content['video'] = {
            'id': video_data.get('id'),
            'url': video_data.get('url'),
            'mime_type': video_data.get('mime_type'),
            'caption': video_data.get('caption'),
        }
    
    elif message_type == 'document':
        doc_data = message_data.get('document', {})
        content['document'] = {
            'id': doc_data.get('id'),
            'url': doc_data.get('url'),
            'filename': doc_data.get('filename'),
            'mime_type': doc_data.get('mime_type'),
        }
    
    return content

