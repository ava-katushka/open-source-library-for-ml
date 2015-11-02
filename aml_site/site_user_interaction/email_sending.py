__author__ = 'ava-katushka'
import smtplib
from email.mime.text import MIMEText
from smtplib import SMTP
from email.mime.text import MIMEText
from email.header import Header
from email.utils import parseaddr, formataddr

"""Send an email.

    All arguments should be Unicode strings (plain ASCII works as well).

    Only the real name part of sender and recipient addresses may contain
    non-ASCII characters.

    The email will be properly MIME encoded and delivered though SMTP to
    localhost port 25.  This is easy to change if you want something different.

    The charset of the email will be the first one out of US-ASCII, ISO-8859-1
    and UTF-8 that can represent all the characters occurring in the email.
"""

def sendMail(body):
    sender = 'aml_site@mail.ru'
    recipient  =  'aml_feedback@mail.ru'
    username =  'aml_site@mail.ru'
    password = 'aml_always'
    subject = 'text-classificator-feedback'
    smtp_server = 'smtp.mail.ru'

    # Header class is smart enough to try US-ASCII, then the charset we
    # provide, then fall back to UTF-8.
    header_charset = 'ISO-8859-1'

    # We must choose the body charset manually
    for body_charset in 'US-ASCII', 'ISO-8859-1', 'UTF-8':
        try:
            body.encode(body_charset)
        except UnicodeError:
            pass
        else:
            break

    # Split real name (which is optional) and email address parts
    sender_name, sender_addr = parseaddr(sender)
    recipient_name, recipient_addr = parseaddr(recipient)

    # We must always pass Unicode strings to Header, otherwise it will
    # use RFC 2047 encoding even on plain ASCII strings.
    sender_name = str(Header(unicode(sender_name), header_charset))
    recipient_name = str(Header(unicode(recipient_name), header_charset))

    # Make sure email addresses do not contain non-ASCII characters
    sender_addr = sender_addr.encode('ascii')
    recipient_addr = recipient_addr.encode('ascii')

    # Create the message ('plain' stands for Content-Type: text/plain)
    msg = MIMEText(body.encode(body_charset), 'plain', body_charset)
    msg['From'] = formataddr((sender_name, sender_addr))
    msg['To'] = formataddr((recipient_name, recipient_addr))
    msg['Subject'] = Header(unicode(subject), header_charset)

    # Send the message via SMTP
    server = smtplib.SMTP(smtp_server)
    server.starttls()
    server.login(username,password)
    server.sendmail(sender, recipient, msg.as_string())
    server.quit()

#sendMail("Check work 2. Hello AML Feedback with Subject!");