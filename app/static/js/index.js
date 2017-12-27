$(function() {
    $('#fileupload').fileupload({
        url: 'upload',
        dataType: 'json',
        add: function(e, data) {
            data.submit();
        },
        success: function(response, status) {
            console.log(response);
        },
        error: function(error) {
            console.log(error);
        }
    });
})
