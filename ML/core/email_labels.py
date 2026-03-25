from enum import Enum

class SpamHamLabel(str, Enum):
    HAM = "ham"
    SPAM = "spam"

class HamIntentLabel(str, Enum):
    JOB_OFFER = "job_offer"
    JOB_UPDATE = "job_update"
    JOB_REJECTION = "job_rejection"
    JOB_ALERTS = "job_alerts"
    VERIFICATION_OTP = "verification_otp"
    ACCOUNT_SECURITY = "account_security"
    FINANCIAL_TRANSACTION = "financial_transaction"
    ORDER_SHIPPING = "order_shipping"
    GOVERNMENT_OFFICIAL = "government_official"
    MEETING_CALENDAR = "meeting_calendar"
    SUPPORT_TICKET = "support_ticket"
    NEWSLETTER_MARKETING = "newsletter_marketing"
    SOCIAL_PERSONAL = "social_personal"
    OTHER_HAM = "other_ham"

HAM_INTENT_DESCRIPTIONS: dict[str, str] = {
    HamIntentLabel.JOB_OFFER.value: "Offer/selection/compensation emails.",
    HamIntentLabel.JOB_UPDATE.value: "Interview/scheduling/recruiter updates.",
    HamIntentLabel.JOB_REJECTION.value: "Regret/rejection/position closed emails.",
    HamIntentLabel.JOB_ALERTS.value: "Job alerts from portals like LinkedIn, Naukri, Indeed.",
    HamIntentLabel.VERIFICATION_OTP.value: "OTP, code verification, email verification.",
    HamIntentLabel.ACCOUNT_SECURITY.value: "Password reset, login alert, security notice.",
    HamIntentLabel.FINANCIAL_TRANSACTION.value: "Bank/payment/receipt/invoice transactions.",
    HamIntentLabel.ORDER_SHIPPING.value: "Purchase confirmation, shipping, delivery updates.",
    HamIntentLabel.GOVERNMENT_OFFICIAL.value: "Official government messages (tax, ID, citizen services).",
    HamIntentLabel.MEETING_CALENDAR.value: "Invites, calendar changes, meeting reminders.",
    HamIntentLabel.SUPPORT_TICKET.value: "Support case opened/updated/resolved.",
    HamIntentLabel.NEWSLETTER_MARKETING.value: "Promotions, newsletters, product campaigns.",
    HamIntentLabel.SOCIAL_PERSONAL.value: "Personal/social communication.",
    HamIntentLabel.OTHER_HAM.value: "Legitimate email that fits no class above.",
}
