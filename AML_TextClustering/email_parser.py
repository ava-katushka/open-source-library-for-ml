
import re
import os
from email.parser import Parser
from email.header import decode_header

def get_subject(email):
    parsed_message = ''
    charset = email.get_content_charset()
    subject = email.get('Subject')
    if subject is not None:
        decoded = decode_header(subject)
        ans = []
        for subj in decoded:
            if len(subj[0]) > 0:
                if subj[1] is not None:
                    parsed_message =  subj[0].decode(subj[1])
                else:
                    try:
                        parsed_message = subj[0].decode(charset)
                    except:
                        parsed_message = ''
                ans.append(parsed_message)
            else:
                ans.append('')
        return '\n'.join(ans)
    else:
        return ''

    
def get_decoded_email_body(msg):
    text = ""
    if msg.is_multipart():
        html = None
        for part in msg.get_payload():
            charset = part.get_content_charset()
            if charset is None:
                charset = 'windows-1251'
            if part.get_content_type() == 'text/plain':
                text = unicode(part.get_payload(decode=True), str(charset), "ignore")

            if part.get_content_type() == 'text/html':
                html = unicode(part.get_payload(decode=True), str(charset), "ignore")

        if text is not None:
            return text.strip()
        elif html is not None:
            return html.strip()
        else:
            return ""
    else:
        chst = msg.get_content_charset()
        if chst is None:
            chst = 'windows-1251'
        text = unicode(msg.get_payload(decode=True), chst, 'ignore')
        return text.strip()    