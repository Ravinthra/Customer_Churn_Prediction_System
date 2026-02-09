from rest_framework import serializers


class ChurnInputSerializer(serializers.Serializer):
    """
    Serializer for churn prediction input validation.
    """
    tenure = serializers.IntegerField(
        min_value=0,
        max_value=1000,
        help_text="Customer tenure in months"
    )
    MonthlyCharges = serializers.FloatField(
        min_value=0,
        max_value=1000000,
        help_text="Monthly charge amount"
    )
    TotalCharges = serializers.FloatField(
        min_value=0,
        max_value=100000000,
        help_text="Total charges to date"
    )
    Contract = serializers.IntegerField(
        min_value=0,
        max_value=2,
        help_text="Contract type: 0=Month-to-month, 1=One year, 2=Two year"
    )
    PaymentMethod = serializers.IntegerField(
        min_value=0,
        max_value=3,
        help_text="Payment method: 0=Electronic check, 1=Mailed check, 2=Bank transfer, 3=Credit card"
    )
    
    def validate(self, data):
        """
        Validate that TotalCharges is reasonable given tenure and MonthlyCharges.
        """
        tenure = data.get('tenure', 0)
        monthly = data.get('MonthlyCharges', 0)
        total = data.get('TotalCharges', 0)
        
        # Basic sanity check: TotalCharges shouldn't be less than MonthlyCharges
        # (unless brand new customer with tenure=0)
        if tenure > 0 and total < monthly:
            raise serializers.ValidationError({
                "TotalCharges": "Total charges should be at least equal to monthly charges for existing customers."
            })
        
        return data
