BUCKET = lnn
FOLDER = deploy
REGION = us-west-2

deploy:
	@[ -d tmp ] || mkdir tmp
	zip tmp/templates.zip master.yaml infrastructure/* services/*
	aws s3 cp tmp/templates.zip s3://$(BUCKET)/$(FOLDER)/ --region $(REGION)
	aws s3 cp master.yaml s3://$(BUCKET)/$(FOLDER)/ --region $(REGION)
	aws s3 cp --recursive infrastructure/ s3://$(BUCKET)/$(FOLDER)/infrastructure --region $(REGION)
	aws s3 cp --recursive services/ s3://$(BUCKET)/$(FOLDER)/services --region $(REGION)

clean:
#	aws s3 rm s3://$(BUCKET)/$(FOLDER)/templates.zip --region $(REGION)
	aws s3 rm --recursive s3://$(BUCKET)/$(FOLDER)
#	aws s3 rm s3://$(BUCKET)/$(FOLDER)/master.yaml --region $(REGION)
#	aws s3 rm --recursive s3://$(BUCKET)/$(FOLDER)/templates --region $(REGION)
	rm -rf tmp

all: deploy

.PHONY: deploy all clean